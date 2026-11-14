import unittest

from tests.runtime.runtime_test_utils import _make_linear_dag, _make_diamond_dag, _make_op
from stratum.optimizer.physical.rewrites._lifetime_planning import plan_input_removals
from stratum.optimizer.logical._base import OperandRef


def _plan_remove_schedule(ops, split_pos, pinned_ops=None):
    """Run plan_input_removals and return the computed remove_after dict."""
    pinned = set(pinned_ops) if pinned_ops else set()
    plan_input_removals(ops, pinned)
    return {op: op.remove_after for op in ops}


# ---------------------------------------------------------------------------
# Last-use release scheduling tests (plan_input_removals).
# ---------------------------------------------------------------------------

class TestRemoveSchedule(unittest.TestCase):

    def test_linear_no_split(self):
        ops = _make_linear_dag()  # A -> B -> C
        schedule = _plan_remove_schedule(ops, split_pos=None)
        # After B processes: A's single consumer (B) is done -> remove A
        self.assertEqual(schedule[ops[1]], [ops[0]])
        # After C processes: B's single consumer (C) is done -> remove B
        self.assertEqual(schedule[ops[2]], [ops[1]])
        # A has no inputs, nothing to remove
        self.assertEqual(schedule[ops[0]], [])

    def test_linear_with_split_pinned(self):
        ops = _make_linear_dag()  # A -> B -> C
        # Split at B (pos=1). A is pinned (feeds B which is post-split).
        schedule = _plan_remove_schedule(ops, split_pos=1, pinned_ops=[ops[0]])
        # After B: A is pinned, so NOT removed
        self.assertEqual(schedule[ops[1]], [])
        # After C: B removed (not pinned, single consumer done)
        self.assertEqual(schedule[ops[2]], [ops[1]])

    def test_diamond_no_split(self):
        a, b, c, d = _make_diamond_dag()  # A -> {B, C} -> D
        ops = [a, b, c, d]
        schedule = _plan_remove_schedule(ops, split_pos=None)
        # A has 2 outputs (B, C). After B: remaining=1, no removal.
        self.assertEqual(schedule[b], [])
        # After C: A has no remaining consumers -> remove A.
        self.assertEqual(schedule[c], [a])
        # After D: B and C both have 1 consumer (D), both removed
        self.assertCountEqual(schedule[d], [b, c])

    def test_diamond_with_split(self):
        a, b, c, d = _make_diamond_dag()
        # Linearized: [A, C, B, D], split_pos=2 (B is split).
        # A feeds B (post-split) and C (pre-split) -> A is pinned.
        # C feeds D (post-split) -> C is pinned.
        ops = [a, c, b, d]
        schedule = _plan_remove_schedule(ops, split_pos=2, pinned_ops=[a, c])
        # After C: A is pinned, not removed
        self.assertEqual(schedule[c], [])
        # After B: A is pinned, not removed
        self.assertEqual(schedule[b], [])
        # After D: B removed (not pinned), C pinned (not removed)
        self.assertEqual(schedule[d], [b])

    def test_complex_dag_with_non_descendant_branches(self):
        """DAG: A -> B (split), A -> C, C -> E, E -> D, B -> D
        Linearized: [A, C, E, B, D], split_pos=3.
        """
        a = _make_op("A")
        b = _make_op("B")
        c = _make_op("C")
        e = _make_op("E")
        d = _make_op("D")

        a.outputs = [b, c]
        b.inputs = [a]
        b.outputs = [d]
        c.inputs = [a]
        c.outputs = [e]
        e.inputs = [c]
        e.outputs = [d]
        d.inputs = [b, e]

        ops = [a, c, e, b, d]
        # A feeds B (post-split) -> pinned. E feeds D (post-split) -> pinned.
        # C only feeds E (pre-split) -> not pinned.
        schedule = _plan_remove_schedule(ops, split_pos=3, pinned_ops=[a, e])
        # After C: A is pinned, not removed
        self.assertEqual(schedule[c], [])
        # After E: C removed (single consumer done, not pinned)
        self.assertEqual(schedule[e], [c])
        # After B: A is pinned, not removed
        self.assertEqual(schedule[b], [])
        # After D: B removed (not pinned), E pinned (not removed)
        self.assertEqual(schedule[d], [b])

    def test_mixed_pre_post_consumers(self):
        """Pre-split op with both pre-split and post-split consumers."""
        a = _make_op("A")
        b = _make_op("B")
        c = _make_op("C")
        a.outputs = [b, c]
        b.inputs = [a]
        c.inputs = [a]
        ops = [a, b, c]
        # Split at pos=2 (C is post-split). A feeds C -> pinned.
        schedule = _plan_remove_schedule(ops, split_pos=2, pinned_ops=[a])
        # After B: A has 2 consumers, remaining=1, not removed (also pinned)
        self.assertEqual(schedule[b], [])
        # After C: A has no remaining consumers but is retained across folds.
        self.assertEqual(schedule[c], [])

    def test_state_is_released_apart_from_data(self):
        """Release a fit operator's data and state independently at their last uses."""
        x = _make_op("X")
        fit = _make_op("T_fit")
        after = _make_op("after_fit")
        predict = _make_op("T_predict")
        sink = _make_op("sink")
        fit.mode, predict.mode = "fit_transform", "transform"
        x.outputs = [fit, predict]
        fit.inputs, fit.outputs = [x], [predict, after]
        after.inputs, after.outputs = [fit], [sink]
        predict.inputs, predict.outputs = [x, fit], [sink]
        predict.state_ref = OperandRef(1)
        sink.inputs = [after, predict]
        ops = [x, fit, predict, after, sink]
        schedule = _plan_remove_schedule(ops, split_pos=None)
        # Prediction is the last state use. Retain the data output for `after`.
        self.assertEqual(schedule[predict], [x])
        self.assertEqual(predict.release_state_after, [fit])
        self.assertEqual(schedule[after], [fit])
        self.assertEqual(after.release_state_after, [])


if __name__ == "__main__":
    unittest.main()
