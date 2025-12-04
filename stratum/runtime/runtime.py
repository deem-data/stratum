import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from skrub._data_ops import DataOp
from skrub._data_ops._evaluation import _Graph
import logging
from types import SimpleNamespace
from stratum.logical_optimizer.optimize import topological_traverse
from skrub._data_ops._data_ops import Value
from skrub._data_ops._choosing import Choice
logger = logging.getLogger(__name__)


def grid_search(dag: DataOp, cv=None, scoring=None, return_predictions=False):
    # dag.skb.draw_graph().open()
    return Runtime(dag, cv, scoring).grid_search(return_predictions)

def evaluate(dag: DataOp, seed: int = 42, test_size = 0.2):
    return Runtime(dag, None, None).evaluate(seed, test_size)

class Runtime:
    def __init__(self, dag: DataOp, cv, scoring):
        self.dag = dag
        self.mode = "fit_transform"
        g = _Graph().run(dag)
        self.nodes, self.parents, self.children = g["nodes"], g["parents"], g["children"]
        self.order = topological_traverse(self.nodes, self.parents, self.children)
        self.cv, self.scoring = cv,scoring
        self.outputs, self.env = {}, {}
        self.full_x, self.full_y = None, None
        self.node_x, self.node_y, self.pos_xy = None, None, None

    def evaluate(self, seed: int = 42, test_size = 0.2):
        self.compute_xy()
        x_train, x_test, y_train, y_test = train_test_split(self.full_x, self.full_y, test_size=test_size, random_state=seed)
        self.compute(self.pos_xy, x_train, y_train)
        pred = self.compute(self.pos_xy, x_test, mode="predict")
        return pred


    def grid_search(self, return_predictions=False):
        # prototype for sequential top-down iteration of the DAG
        if logger.isEnabledFor(logging.DEBUG):
            self.dag.skb.draw_graph().open()

        # start with computing till X and y node
        self.compute_xy()
        results, predictions = [], []

        if self.cv is None:
            self.cv = KFold(n_splits=5, shuffle=True, random_state=42)

        for i, (train_index, test_index) in enumerate(self.cv.split(self.outputs[self.node_x])):
            logger.debug(f"CV Fold Nr. {i+1}")
            x_train, x_test = self.full_x.iloc[train_index], self.full_x.iloc[test_index]
            y_train, y_test = self.full_y .iloc[train_index], self.full_y .iloc[test_index]

            # fit and predict the pipeline
            self.compute(self.pos_xy, x_train, y_train)
            df = self.compute(self.pos_xy, x_test, mode="predict")
            if return_predictions:
                predictions.append(df.copy())

            # scoring
            df["scores"] = df["vals"].apply(
                lambda a: mean_squared_error(y_test, a))*(-1 if self.scoring == "neg_mean_squared_error" else 1)
            df = df.drop("vals", axis=1)
            results.append(df)

        results = pd.concat(results, axis=0)
        results = results.groupby("id").aggregate("mean").sort_values(by="scores", ascending=False)

        return results, predictions if return_predictions else results

    def compute(self, start_pos: int, x, y = None, mode="fit_transform"):
        self.mode = mode
        self.outputs[self.node_x] = x
        self.outputs[self.node_y] = y
        for node in self.order[(start_pos + 1):]:
            self.outputs[node] = self.process_op(self.nodes[node], node)

        return pd.DataFrame(self.outputs[self.order[-1]]) if mode == "predict" else None

    def compute_xy(self):
        # iterate till we find the X and y nodes
        pos_x, pos_y = None, None
        for i, node in enumerate(self.order):
            current_op = self.nodes[node]
            self.outputs[node]  = self.process_op(current_op, node)
            if current_op.skb.is_X:
                pos_x, self.node_x = i, node
            elif current_op.skb.is_y:
                pos_y, self.node_y,  = i, node
            if pos_x is not None and pos_y is not None:
                break
        self.full_x, self.full_y = self.outputs[self.node_x], self.outputs[self.node_y]
        self.pos_xy = max(pos_x, pos_y)

    def process_op(self, dataop: DataOp, node: int):
        impl = dataop._skrub_impl
        if isinstance(impl, Value) and isinstance(impl.value, Choice):
            choice = impl.value
            results = []
            child_iter = iter(self.children.get(node,[]))
            for name, outcome in zip(choice.outcome_names , choice.outcomes):
                if isinstance(outcome, DataOp):
                    results.append({ "id" : name, "vals" : self.outputs[next(child_iter)]})
                else:
                    results.append({"id" : name, "vals" : outcome})
            current_output = results[0] if len(results) == 1 else results
        elif hasattr(impl, "eval"):
            last_yield = None
            gen = impl.eval(mode=self.mode, environment=self.env)
            child_iter = iter(self.children.get(node,[]))
            while True:
                try:
                    last_yield = gen.send(last_yield)
                except StopIteration as e:
                    current_output = e.value
                    break
                if isinstance(last_yield, DataOp):
                    last_yield = self.outputs[next(child_iter)]
        else:
            try:
                fields = self.replace_fields_with_values(impl, children=self.children.get(node,[]))
                ns = SimpleNamespace(**{k:v for k,v in fields})
                current_output = impl.compute(ns, self.mode, self.env)
            except Exception as e:
                raise Exception(f"Error processing implementation '{impl}' [Node {node}]: {e}")
        return current_output

    def replace_fields_with_values(self, impl, children):
        child_iter = iter(children)

        def replace_dataop(value):
            """Recursively replace DataOp instances with their outputs."""
            if isinstance(value, DataOp):
                return self.outputs[next(child_iter)]
            elif isinstance(value, (list, tuple)):
                new_seq = [replace_dataop(item) for item in value]
                return type(value)(new_seq)
            elif isinstance(value, dict):
                return {key: replace_dataop(val) for key, val in value.items()}
            else:
                return value

        return [(field, replace_dataop(getattr(impl, field))) for field in impl._fields]