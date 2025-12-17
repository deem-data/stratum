from time import perf_counter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, check_cv
from skrub._data_ops._data_ops import EvalMode
from skrub._data_ops import DataOp
from types import SimpleNamespace
import io
from contextlib import redirect_stdout, redirect_stderr
from stratum._config import FLAGS
from stratum.logical_optimizer._optimize import optimize,OptConfig
from stratum.logical_optimizer._ops import ChoiceOp, Op, ValueOp

import pandas as pd
import logging
logger = logging.getLogger(__name__)


def grid_search(dag: DataOp, cv=None, scoring=None, return_predictions=False, show_stats=False):
    """Perform grid search with cross-validation on a DataOp DAG."""
    # TODO maybe remove the option to show stats here and make args similar to scikit-learn's grid_search
    show_stats = FLAGS.stats or show_stats
    ops_ordered = optimize(dag, OptConfig(cse=True))
    sched = Scheduler(ops_ordered, show_stats)

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

def evaluate(dag: DataOp, seed: int = 42, test_size = 0.2, cse: bool = False):
    """Evaluate a DataOp DAG with train/test split."""
    ops_ordered = optimize(dag, OptConfig(cse=cse))
    return Scheduler(ops_ordered).evaluate(seed, test_size)

class Scheduler:
    """Scheduler for executing DataOpDAGs in topological order."""
    
    def __init__(self, ops_ordered: list[Op], print_heavy_hitters=False):
        """Initialize scheduler with a data operations DAG."""
        self.ops_ordered = ops_ordered
        self.mode = "fit_transform"
        self.intermediates, self.env = {}, {}
        self.flagged_for_recomputation = []
        self.full_x, self.full_y, self.op_x, self.op_y = None, None, None, None
        self.pos_x, self.pos_y, self.pos_xy = None, None, None
        self.timings = [] if print_heavy_hitters else None

    def evaluate(self, seed: int = 42, test_size = 0.2):
        """Evaluate the pipeline with a train/test split and return predictions."""
        try:
            self.compute_xy()
        except RuntimeError as e:
            if "X and y nodes not found in the DAG" in str(e):
                logger.warning("X and y nodes not found in the DAG, returning the last node")
                return self.ops_ordered[-1].intermediate
            else:
                raise e
        x_train, x_test, y_train, y_test = train_test_split(self.full_x, self.full_y, test_size=test_size, random_state=seed)
        self.compute(self.pos_xy, x_train, y_train)
        pred = self.compute(self.pos_xy, x_test, mode="predict")
        return pred["vals"][0]


    def grid_search(self, cv=None, scoring=None, return_predictions=False):
        """Perform grid search with cross-validation on the DataOp DAG in a sequential top-down manner."""

        # start with computing till X and y node
        logger.debug("\n", "="*100, "\n", "Starting grid search", "\n","="*100, "\n")
        self.compute_xy()
        results, predictions = [], []

        # default to scikit-learn's CV
        cv = check_cv(cv)
        logger.debug("\n", "="*100, "\n", "XY computed", "\n","="*100, "\n")
        #TODO we can parallelize over the folds
        for i, (train_index, test_index) in enumerate(cv.split(self.full_x)):
            logger.debug(f"CV Fold Nr. {i+1}")
            x_train, x_test = self.full_x.iloc[train_index], self.full_x.iloc[test_index]
            y_train, y_test = self.full_y .iloc[train_index], self.full_y .iloc[test_index]

            # fit and predict the pipeline
            self.compute(self.pos_xy, x_train, y_train)
            logger.debug("\n", "="*100, "\n", "Training done for fold", i+1, "\n","="*100, "\n")
            df = self.compute(self.pos_xy, x_test, mode="predict")
            logger.debug("\n", "="*100, "\n", "Predicting done for fold", i+1, "\n","="*100, "\n")
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
        ops_to_compute = self.ops_ordered[(start_pos + 1):]
        if len(self.flagged_for_recomputation) != 0:
            ops_to_compute = self.flagged_for_recomputation + ops_to_compute
        self.mode = mode
        self.op_x.intermediate = x
        self.op_y.intermediate = y

        for node in ops_to_compute:
            self.process_op(node)

        if mode == "predict":
            pred = self.ops_ordered[-1].intermediate
            if isinstance(pred, list):
                return pd.DataFrame(pred)
            else:
                return pd.DataFrame({"vals": [pred], "id": ["default"]})
        return None

    def compute_xy(self):
        """Compute nodes until X and y nodes are found and store them."""
        for i, op in enumerate(self.ops_ordered):
            self.process_op(op)
            if isinstance(op.skrub_impl, EvalMode):
                self.flagged_for_recomputation.append(op)
            # TODO Nodes between X and y, might be recomputed as well
            if op.is_X:
                self.pos_x, self.op_x, self.full_x = i, op, op.intermediate
            elif op.is_y:
                self.pos_y, self.op_y, self.full_y = i, op, op.intermediate
            if self.pos_x is not None and self.pos_y is not None:
                break
        if self.pos_x is None or self.pos_y is None:
            raise RuntimeError("X and y nodes not found in the DAG")
        self.pos_xy = max(self.pos_x, self.pos_y)

    def process_op(self, op: Op):
        """Process a single DataOp node and return its output."""
        impl = op.skrub_impl
        t0 = perf_counter() if self.timings is not None else 0
                
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            if isinstance(op, ValueOp):
                op.intermediate = op.value
            elif isinstance(op, ChoiceOp):
                results = []
                outcome_iter = iter(op.parents)
                for name in op.make_outcome_names():
                    results.append({ "id" : name, "vals" : next(outcome_iter).intermediate})
                op.intermediate = results[0] if len(results) == 1 else results
            elif hasattr(impl, "eval"):
                # DataOp with eval method have a fused implementation of the generator and the compute method
                # we need to iterate over the generator and replace the requested fields with correct inputs
                last_yield = None
                gen = impl.eval(mode=self.mode, environment=self.env)
                parent_iter = iter(op.parents)
                while True:
                    try:
                        last_yield = gen.send(last_yield)
                    except StopIteration as e:
                        op.intermediate = e.value
                        # if self.mode == "predict" and isinstance(impl.estimator, TableVectorizer):
                        #     op.intermediate = "test"

                        break
                    if isinstance(last_yield, DataOp):
                        last_yield = next(parent_iter).intermediate
            else:
                try:
                    fields = self.replace_fields_with_values(impl, parents=op.parents)
                    ns = SimpleNamespace(**{k:v for k,v in fields})
                    op.intermediate = impl.compute(ns, self.mode, self.env)
                except Exception as e:
                    raise RuntimeError(f"Error processing Op '{op.name}': {e}")
        
        # Store captured output in the op
        op.output = {
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue()
        }
        
        if self.timings is not None:
            duration = perf_counter() - t0
            self.timings.append((op.name, duration))
        return op

    def replace_fields_with_values(self, impl, parents):
        """Replace DataOp fields in implementation with their computed values."""
        parent_iter = iter(parents)

        def replace_dataop(value):
            """Recursively replace DataOp instances with their actual values."""
            if isinstance(value, DataOp):
                return next(parent_iter).intermediate
            elif isinstance(value, (list, tuple)):
                new_seq = [replace_dataop(item) for item in value]
                return type(value)(new_seq)
            elif isinstance(value, dict):
                return {key: replace_dataop(val) for key, val in value.items()}
            else:
                return value

        return [(field, replace_dataop(getattr(impl, field))) for field in impl._fields]