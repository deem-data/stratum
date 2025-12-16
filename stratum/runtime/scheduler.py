from time import perf_counter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, check_cv
from skrub._data_ops import DataOp
from skrub._data_ops._evaluation import _Graph
from types import SimpleNamespace
from stratum.logical_optimizer.optimize import topological_traverse
from stratum._config import FLAGS
from skrub._data_ops._data_ops import Value, EvalMode
from skrub._data_ops._choosing import Choice

import pandas as pd
import logging
logger = logging.getLogger(__name__)


def grid_search(dag: DataOp, cv=None, scoring=None, return_predictions=False, show_stats=False):
    """Perform grid search with cross-validation on a DataOp DAG."""
    # TODO maybe remove the option to show stats here and make args similar to scikit-learn's grid_search
    show_stats = FLAGS.stratum_stats or show_stats
    sched = Scheduler(dag, show_stats)
    out = sched.grid_search(cv, scoring, return_predictions)
    
    # Heavy hitters
    if show_stats:
        table = pd.DataFrame(sched.timings, columns=["Op", "time"])
        table = table.groupby("Op").aggregate(["sum", "count"])
        table.columns = ["Time", "Count"]
        table = table.reset_index().sort_values(by="Time", ascending=False)
        print("\n" + "=" * 80)
        print(f"Heavy hitters (sorted by time spent in DataOp evaluation):")
        print(table.head(20).to_string(index=False))
        print("=" * 80 + "\n")
    return out

def evaluate(dag: DataOp, seed: int = 42, test_size = 0.2):
    """Evaluate a DataOp DAG with train/test split."""
    return Scheduler(dag).evaluate(seed, test_size)

class Scheduler:
    """Scheduler for executing DataOpDAGs in topological order."""
    
    def __init__(self, dag: DataOp, print_heavy_hitters=False):
        """Initialize scheduler with a data operations DAG."""
        self.dag = dag
        self.mode = "fit_transform"
        g = _Graph().run(dag)
        self.nodes, self.parents, self.children = g["nodes"], g["parents"], g["children"]
        self.order = topological_traverse(self.nodes, self.parents, self.children)
        self.intermediates, self.env = {}, {}
        self.flagged_for_recomputation = []
        self.full_x, self.full_y = None, None
        self.node_x, self.node_y, self.pos_xy = None, None, None
        self.timings = [] if print_heavy_hitters else None

    def evaluate(self, seed: int = 42, test_size = 0.2):
        """Evaluate the pipeline with a train/test split and return predictions."""
        self.compute_xy()
        x_train, x_test, y_train, y_test = train_test_split(self.full_x, self.full_y, test_size=test_size, random_state=seed)
        self.compute(self.pos_xy, x_train, y_train)
        pred = self.compute(self.pos_xy, x_test, mode="predict")
        return pred["vals"][0]


    def grid_search(self, cv=None, scoring=None, return_predictions=False):
        """Perform grid search with cross-validation on the DataOp DAG in a sequential top-down manner."""

        # start with computing till X and y node
        self.compute_xy()
        results, predictions = [], []

        # default to scikit-learn's CV
        cv = check_cv(cv)

        #TODO we can parallelize over the folds
        for i, (train_index, test_index) in enumerate(cv.split(self.intermediates[self.node_x])):
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
                lambda a: mean_squared_error(y_test, a))*(-1 if scoring == "neg_mean_squared_error" else 1)
            df = df.drop("vals", axis=1)
            results.append(df)

        results = pd.concat(results, axis=0)
        results = results.groupby("id").aggregate("mean").sort_values(by="scores", ascending=False)

        return (results, predictions) if return_predictions else results

    def compute(self, start_pos: int, x, y = None, mode="fit_transform"):
        """Compute the pipeline from start_pos onwards with given inputs."""
        if len(self.flagged_for_recomputation) == 0:
            nodes = self.order[(start_pos + 1):]
        else:
            nodes = self.flagged_for_recomputation + self.order[(start_pos + 1):]
        self.mode = mode
        self.intermediates[self.node_x] = x
        self.intermediates[self.node_y] = y
        for node in nodes:
            self.intermediates[node] = self.process_op(self.nodes[node], node)

        if mode == "predict":
            pred = self.intermediates[self.order[-1]]
            if isinstance(pred, list):
                return pd.DataFrame(pred)
            else:
                return pd.DataFrame({"vals": [pred], "id": ["default"]})
        return None

    def compute_xy(self):
        """Compute nodes until X and y nodes are found and store them."""
        pos_x, pos_y = None, None
        for i, node in enumerate(self.order):
            current_op = self.nodes[node]
            self.intermediates[node] = self.process_op(current_op, node)
            if isinstance(current_op._skrub_impl, EvalMode):
                self.flagged_for_recomputation.append(node)
            # TODO Nodes between X and y, might be recomputed as well
            if current_op.skb.is_X:
                pos_x, self.node_x = i, node
            elif current_op.skb.is_y:
                pos_y, self.node_y,  = i, node
            if pos_x is not None and pos_y is not None:
                break
        if self.node_x is None or self.node_y is None:
            raise RuntimeError("X and y nodes not found in the DAG")
        self.full_x, self.full_y = self.intermediates[self.node_x], self.intermediates[self.node_y]
        self.pos_xy = max(pos_x, pos_y)

    def process_op(self, dataop: DataOp, node: int):
        """Process a single DataOp node and return its output."""
        impl = dataop._skrub_impl
        t0 = perf_counter() if self.timings is not None else 0
        if isinstance(impl, Value) and isinstance(impl.value, Choice):
            choice = impl.value
            results = []
            child_iter = iter(self.children.get(node,[]))
            outcome_names = choice.outcome_names or [f"option{i}" for i in range(len(choice.outcomes))]
            for name, outcome in zip(outcome_names, choice.outcomes):
                if isinstance(outcome, DataOp):
                    results.append({ "id" : name, "vals" : self.intermediates[next(child_iter)]})
                else:
                    raise NotImplementedError("Choices with non-DataOp outcomes are not supported yet.")
            current_output = results[0] if len(results) == 1 else results
        elif hasattr(impl, "eval"):
            # DataOp with eval method have a fused implementation of the generator and the compute method
            # we need to iterate over the generator and replace the requested fields with correct inputs
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
                    last_yield = self.intermediates[next(child_iter)]
        else:
            try:
                fields = self.replace_fields_with_values(impl, children=self.children.get(node,[]))
                ns = SimpleNamespace(**{k:v for k,v in fields})
                current_output = impl.compute(ns, self.mode, self.env)
            except Exception as e:
                raise RuntimeError(f"Error processing implementation '{impl}' [Node {node}]: {e}")
        if self.timings is not None:
            duration = perf_counter() - t0
            self.timings.append((self.nodes[node].__skrub_short_repr__(), duration))
        return current_output

    def replace_fields_with_values(self, impl, children):
        """Replace DataOp fields in implementation with their computed values."""
        child_iter = iter(children)

        def replace_dataop(value):
            """Recursively replace DataOp instances with their actual values."""
            if isinstance(value, DataOp):
                return self.intermediates[next(child_iter)]
            elif isinstance(value, (list, tuple)):
                new_seq = [replace_dataop(item) for item in value]
                return type(value)(new_seq)
            elif isinstance(value, dict):
                return {key: replace_dataop(val) for key, val in value.items()}
            else:
                return value

        return [(field, replace_dataop(getattr(impl, field))) for field in impl._fields]