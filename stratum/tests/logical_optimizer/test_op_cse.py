import unittest

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import stratum as st
from stratum.optimizer import apply_op_cse
from stratum.optimizer._optimize import convert_to_ops
from stratum.optimizer._op_utils import topological_iterator, validate_dag
from stratum.optimizer.ir._ops import ChoiceOp


def pre_process(df):
    return df


def _count_ops(root):
    return sum(1 for _ in topological_iterator(root))


class TestOpCSE(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "datetime": [
                "2025-11-01 10:00:00",
                "2025-11-02 15:30:00",
                "2025-11-03 09:45:00",
            ],
        })

    def test_dedup_binop(self):
        # t3 and t4 are the same subexpression (t1 + t2); they must collapse.
        t1 = st.as_data_op(1)
        t2 = st.as_data_op(2)
        t3 = t1 + t2
        t4 = t1 + t2
        dag = t3 + t4

        root = convert_to_ops(dag)
        before = _count_ops(root)
        root = apply_op_cse(root)
        after = _count_ops(root)

        # ValueOp(1), ValueOp(2), add(t1,t2) x2, add(t3,t4)  ->  one add(t1,t2) survives
        self.assertEqual(before - after, 1)
        validate_dag(root)

    def test_operand_refs_renumbered_on_collapse(self):
        # After CSE, the outer add consumes the *same* inner add twice: it must
        # collapse to a single input edge with both operands re-pointed to index 0.
        t1 = st.as_data_op(1)
        t2 = st.as_data_op(2)
        inner_a = t1 + t2
        inner_b = t1 + t2
        dag = (inner_a + inner_b)

        root = apply_op_cse(convert_to_ops(dag))
        validate_dag(root)  # would raise if an OperandRef pointed out of range

        outer = root
        # the two equal inner adds collapse to one input edge...
        self.assertEqual(len(outer.inputs), 1)
        # ...and both operands of the outer add reference that single edge (index 0)
        self.assertEqual(outer.left.k, 0)
        self.assertEqual(outer.right.k, 0)

    def test_idempotent(self):
        t1 = st.as_data_op(1)
        t2 = st.as_data_op(2)
        dag = ((t1 + t2) + (t1 + t2))

        root = apply_op_cse(convert_to_ops(dag))
        first = _count_ops(root)
        root = apply_op_cse(root)
        second = _count_ops(root)
        self.assertEqual(first, second)
        validate_dag(root)

    def test_shared_dataframe_pipeline(self):
        # Two identical feature-engineering chains over the same X must be shared.
        data = st.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()

        X1 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format="%Y-%m-%d %H:%M:%S"))
        X1B = X1.assign(year=X1["datetime"].dt.year, month=X1["datetime"].dt.month)

        X2 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format="%Y-%m-%d %H:%M:%S"))
        X2B = X2.assign(year=X2["datetime"].dt.year, month=X2["datetime"].dt.month)

        out = st.choose_from({"a": X1B, "b": X2B}).as_data_op()

        root = convert_to_ops(out)
        before = _count_ops(root)
        root = apply_op_cse(root)
        after = _count_ops(root)
        self.assertLess(after, before)
        validate_dag(root)

    def test_distinct_estimators_not_merged(self):
        data = st.var("data", self.df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        t1 = X.skb.apply_func(pre_process)
        y1 = t1.skb.apply(RandomForestRegressor(random_state=42), y=y)
        t2 = X.skb.apply_func(pre_process)
        y2 = t2.skb.apply(RandomForestRegressor(random_state=123), y=y)
        out = st.choose_from({"a": y1, "b": y2}).as_data_op()

        root = apply_op_cse(convert_to_ops(out))
        validate_dag(root)
        # Two estimators differ in random_state -> both must remain.
        estimators = [
            op for op in topological_iterator(root)
            if op.__class__.__name__ in ("EstimatorOp", "TransformerOp")
        ]
        self.assertEqual(len(estimators), 2)

    def test_identical_choice_outcomes_kept_distinct(self):
        # Two structurally-identical choice outcomes must NOT be collapsed: a
        # ChoiceOp addresses its outcomes positionally, so merging them would
        # desync inputs from outcome_names. Internal shared nodes still merge.
        data = st.var("data", self.df)
        X = data[["x"]].skb.mark_as_X()
        a = X.assign(s=X["x"] + 1)
        b = X.assign(s=X["x"] + 1)
        out = st.choose_from({"a": a, "b": b}).as_data_op()

        root = convert_to_ops(out)
        before = _count_ops(root)
        root = apply_op_cse(root)
        after = _count_ops(root)

        choice = next(op for op in topological_iterator(root) if isinstance(op, ChoiceOp))
        self.assertEqual(len(choice.inputs), len(choice.outcome_names))
        self.assertEqual(len(choice.inputs), 2)
        # internal sharing (e.g. the duplicated X["x"] + 1) still collapses
        self.assertLess(after, before)
        validate_dag(root)

    def test_shared_prefix_across_choice(self):
        # The shared apply_func chain should merge, but the two choice branches
        # (different estimators) must both survive.
        data = st.var("data", self.df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        t1 = X.skb.apply_func(pre_process)
        y1 = t1.skb.apply(RandomForestRegressor(random_state=42), y=y)
        t2 = X.skb.apply_func(pre_process)
        y2 = t2.skb.apply(RandomForestRegressor(random_state=123), y=y)
        out = st.choose_from({"a": y1, "b": y2}).as_data_op()

        root = convert_to_ops(out)
        before = _count_ops(root)
        root = apply_op_cse(root)
        after = _count_ops(root)
        # apply_func(pre_process) over the same X is shared across branches.
        self.assertLess(after, before)
        validate_dag(root)

    def test_dedup_rename(self):
        # t3 and t4 are the same subexpression (t1 + t2); they must collapse.
        t1 = st.as_data_op(pd.DataFrame({"x": [1, 2, 3]}))
        t2 = t1.rename(columns={"x": "y"})
        t3 = t1.rename(columns={"x": "y"})
        t4 = t2.skb.concat([t3])

        root = convert_to_ops(t4)
        before = _count_ops(root)
        root = apply_op_cse(root)
        after = _count_ops(root)

        # ValueOp(1), ValueOp(2), add(t1,t2) x2, add(t3,t4)  ->  one add(t1,t2) survives
        self.assertEqual(before - after, 1)
        validate_dag(root)

    def test_dedup_udf_with_set_arg(self):
        t1 = st.as_data_op(1)
        udf = lambda x, a: x
        t2 = t1.skb.apply_func(udf, a={1, 2, 3})
        t3 = t1.skb.apply_func(udf, a={1, 2, 3})
        t4 = t3 + t2

        root = convert_to_ops(t4)
        before = _count_ops(root)
        root = apply_op_cse(root)
        after = _count_ops(root)

        self.assertEqual(before - after, 1)
        validate_dag(root)


    def test_dedup_udf_list(self):
        t1 = st.as_data_op(1)
        t2 = t1 + 2
        t3 = t1 + 2
        t4 = t1 + 2
        t5 = t1 + 2
        udf = lambda x,l: x + sum(l)
        t6 = t5.skb.apply_func(udf, l=[t2, t3, t4])


        root = convert_to_ops(t6)
        before = _count_ops(root)
        root = apply_op_cse(root)
        after = _count_ops(root)

        self.assertEqual(3, before - after,)
        validate_dag(root)

if __name__ == "__main__":
    unittest.main()
