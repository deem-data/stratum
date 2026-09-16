from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._ops import ValueOp, VariableOp
from stratum.runtime._buffer_pool import BufferPool
import stratum as st
import pandas as pd
import unittest

# dummy function
def pre_process(df):
    return df

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "datetime": [
                "2025-11-01 10:00:00",
                "2025-11-02 15:30:00",
                "2025-11-03 09:45:00"
            ]
        })

    def test_optimize1(self):
        data = st.var("data", self.df).skb.subsample(3)
        X = data[["x", "datetime"]].skb.mark_as_X()

        X1 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2 = X1.assign(
            year=X1["datetime"].dt.year,
            month=X1["datetime"].dt.month)
        out, *_ = optimize(X2, OptConfig(cse=True))
        self.assertIs(out[0].outputs[0], out[1])
        self.assertEqual(len(out[0].inputs), 0)

    def test_optimize2(self):
        data = st.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()

        X1 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2 = X1.assign(
            year=X1["datetime"].dt.year,
            month=X1["datetime"].dt.month)
        config = OptConfig(cse=False, algebraic_rewrites=False, numeric_ops=False, dataframe_ops=False, unroll_choices=False)
        out, *_ = optimize(X2, config)
        self.assertEqual(len(out), 10)
        
    def test_more_ops(self):
        data = st.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        X1 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2 = X1.assign(
            year=X1["datetime"].dt.year,
            month=X1["datetime"].dt.month)
        optimize(X2, OptConfig(cse=True))




class TestResolveConstants(unittest.TestCase):
    """`optimize(dag, env=...)` resolves variables to compile-time constants."""

    # No rewrites, so the converted leaves survive verbatim for inspection.
    _NO_REWRITES = OptConfig(cse=False, algebraic_rewrites=False, numeric_ops=False,
                             dataframe_ops=False, unroll_choices=False)

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def test_variable_resolved_to_constant_with_env(self):
        data = st.var("data", self.df)
        out, *_ = optimize(data[["x"]], self._NO_REWRITES, env={"data": self.df})
        self.assertFalse(any(isinstance(o, VariableOp) for o in out))
        value_ops = [o for o in out if isinstance(o, ValueOp)]
        self.assertEqual(1, len(value_ops))
        pd.testing.assert_frame_equal(value_ops[0].value, self.df)

    def test_variable_kept_without_env(self):
        data = st.var("data", self.df)
        out, *_ = optimize(data[["x"]], self._NO_REWRITES)
        self.assertTrue(any(isinstance(o, VariableOp) for o in out))
        self.assertFalse(any(isinstance(o, ValueOp) for o in out))

    def test_unbound_variable_stays_a_variable(self):
        # env that doesn't bind the variable name leaves it as a VariableOp.
        data = st.var("data", self.df)
        out, *_ = optimize(data[["x"]], self._NO_REWRITES, env={"other": 1})
        self.assertTrue(any(isinstance(o, VariableOp) for o in out))

    def test_resolved_plan_runs_without_environment(self):
        # The whole point: once resolved, the plan executes with an empty env.
        data = st.var("data", self.df)
        out, *_ = optimize(data[["x"]], self._NO_REWRITES, env={"data": self.df})
        pool = BufferPool()
        for op in out:
            inputs = [pool.pin(k) for k in op.inputs]
            pool.put(op, op.process("fit_transform", inputs))
        result = pool.pin(out[-1])
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), self.df[["x"]])


if __name__ == '__main__':
    unittest.main()
