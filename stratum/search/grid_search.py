import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from skrub._data_ops import DataOp
from skrub._data_ops._evaluation import _Graph
import logging
from types import SimpleNamespace
from stratum.logical_optimizer.optimize import topological_traverse
from skrub._data_ops._data_ops import Value
from skrub._data_ops._choosing import Choice
logger = logging.getLogger(__name__)

class GridSearch:
    def __init__(self, dag: DataOp, cv, scoring):
        self.dag = dag
        self.graph = _Graph().run(dag)
        self.nodes = self.graph["nodes"]
        self.parents = self.graph["parents"]
        self.children = self.graph["children"]
        self.order = topological_traverse(self.nodes, self.parents, self.children)
        self.outputs = {}
        self.mode = "fit_transform"
        self.env = {}
        self.cv = cv
        self.scoring = scoring
    
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

    def grid_search(self):
        if logger.isEnabledFor(logging.DEBUG):
            self.dag.skb.draw_graph().open()

        # prototype sequential top down iteration of the DAG

        # iterate till we find the X and y nodes
        position_of_X = None
        position_of_y = None
        node_id_X = None
        node_id_y = None
        for i,node in enumerate(self.order):
            current_op = self.nodes[node]
            current_output = self.process_op(current_op, node)
            self.outputs[node] = current_output
            if current_op.skb.is_X:
                position_of_X = i
                node_id_X = node
            elif current_op.skb.is_y:
                node_id_y = node
                position_of_y = i
            if position_of_X is not None and position_of_y is not None:
                break
        
        if self.cv is None:
            self.cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
        position_after = max(position_of_X, position_of_y)

        original_X = self.outputs[node_id_X]
        original_y = self.outputs[node_id_y]

        df_out = []
        
        for i, (train_index, test_index) in enumerate(self.cv.split(self.outputs[node_id_X])):
            logger.debug(f"CV Fold Nr. {i+1}")
            X_train = self.outputs[node_id_X].iloc[train_index]
            X_test = self.outputs[node_id_X].iloc[test_index]
            y_train = self.outputs[node_id_y].iloc[train_index]
            y_test = self.outputs[node_id_y].iloc[test_index]

            # fit the pipeline
            self.mode = "fit_transform"
            self.outputs[node_id_X] = X_train
            self.outputs[node_id_y] = y_train
            for node in self.order[(position_after+1):]:
                self.outputs[node] = self.process_op(self.nodes[node], node)
            
            
            # predict the pipeline
            self.mode = "predict"
            self.outputs[node_id_X] = X_test
            self.outputs[node_id_y] = None # we don't need the y for prediction
            for node in self.order[(position_after+1):]:
                self.outputs[node] = self.process_op(self.nodes[node], node)

            df = pd.DataFrame(self.outputs[node])
            weight = 1
            if self.scoring == "neg_mean_squared_error":
                weight = -1
            df["scores"] = df["vals"].apply(lambda a: mean_squared_error(y_test, a))*weight
            df = df.drop("vals", axis=1)
            df_out.append(df)

            # reset the outputs for the next fold
            self.outputs[node_id_X] = original_X
            self.outputs[node_id_y] = original_y
        df_out = pd.concat(df_out, axis=0)
        df_out = df_out.groupby("id").aggregate("mean").sort_values(by="scores", ascending=False)
        return df_out

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


def grid_search(dag: DataOp, cv=None, scoring=None):
    # dag.skb.draw_graph().open()
    search = GridSearch(dag, cv, scoring).grid_search()
    return search
