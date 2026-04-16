import unittest

from stratum.tests.runtime.runtime_test_utils import _make_linear_dag, _make_diamond_dag, _make_op
from stratum.optimizer._input_release_planning import plan_input_releases


def _plan_release_schedule(ops, split_pos, flagged_ops, pinned_ops=None):
    """Run plan_input_releases and return the computed release_after dict."""
    pinned = set(pinned_ops) if pinned_ops else set()
    plan_input_releases(ops, pinned)
    return {op: op.release_after for op in ops}


# ---------------------------------------------------------------------------
# Release schedule tests (exercises Scheduler.plan)
# ---------------------------------------------------------------------------

class TestReleaseSchedule(unittest.TestCase):

    def test_linear_no_split(self):
        ops = _make_linear_dag()  # A -> B -> C
        schedule = _plan_release_schedule(ops, split_pos=None, flagged_ops=[])
        # After B processes: A's single consumer (B) is done -> release A
        self.assertEqual(schedule[ops[1]], [ops[0]])
        # After C processes: B's single consumer (C) is done -> release B
        self.assertEqual(schedule[ops[2]], [ops[1]])
        # A has no inputs, nothing to release
        self.assertEqual(schedule[ops[0]], [])

    def test_linear_with_split_pinned(self):
        ops = _make_linear_dag()  # A -> B -> C
        # Split at B (pos=1). A is pinned (feeds B which is post-split).
        schedule = _plan_release_schedule(ops, split_pos=1, flagged_ops=[], pinned_ops=[ops[0]])
        # After B: A is pinned, so NOT released
        self.assertEqual(schedule[ops[1]], [])
        # After C: B released (not pinned, single consumer done)
        self.assertEqual(schedule[ops[2]], [ops[1]])

    def test_diamond_no_split(self):
        a, b, c, d = _make_diamond_dag()  # A -> {B, C} -> D
        ops = [a, b, c, d]
        schedule = _plan_release_schedule(ops, split_pos=None, flagged_ops=[])
        # A has 2 outputs (B, C). After B: remaining=1, no release.
        self.assertEqual(schedule[b], [])
        # After C: A's remaining hits 0 -> release A
        self.assertEqual(schedule[c], [a])
        # After D: B and C both have 1 consumer (D), both released
        self.assertCountEqual(schedule[d], [b, c])

    def test_diamond_with_split(self):
        a, b, c, d = _make_diamond_dag()
        # Linearized: [A, C, B, D], split_pos=2 (B is split).
        # A feeds B (post-split) and C (pre-split) -> A is pinned.
        # C feeds D (post-split) -> C is pinned.
        ops = [a, c, b, d]
        schedule = _plan_release_schedule(ops, split_pos=2, flagged_ops=[], pinned_ops=[a, c])
        # After C: A is pinned, not released
        self.assertEqual(schedule[c], [])
        # After B: A is pinned, not released
        self.assertEqual(schedule[b], [])
        # After D: B released (not pinned), C pinned (not released)
        self.assertEqual(schedule[d], [b])

    def test_flagged_ops_not_pinned(self):
        ops = _make_linear_dag()  # A, B, C
        # Split at C (pos=2), A is flagged (re-executed, not pinned).
        # B feeds C (post-split) -> B is pinned.
        schedule = _plan_release_schedule(ops, split_pos=2, flagged_ops=[ops[0]], pinned_ops=[ops[1]])
        # After B: A released (not pinned, single consumer done)
        self.assertEqual(schedule[ops[1]], [ops[0]])
        # After C: B is pinned, not released
        self.assertEqual(schedule[ops[2]], [])

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
        schedule = _plan_release_schedule(ops, split_pos=3, flagged_ops=[], pinned_ops=[a, e])
        # After C: A is pinned, not released
        self.assertEqual(schedule[c], [])
        # After E: C released (single consumer done, not pinned)
        self.assertEqual(schedule[e], [c])
        # After B: A is pinned, not released
        self.assertEqual(schedule[b], [])
        # After D: B released (not pinned), E pinned (not released)
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
        schedule = _plan_release_schedule(ops, split_pos=2, flagged_ops=[], pinned_ops=[a])
        # After B: A has 2 consumers, remaining=1, not released (also pinned)
        self.assertEqual(schedule[b], [])
        # After C: A's remaining hits 0 but pinned -> not released
        self.assertEqual(schedule[c], [])


if __name__ == "__main__":
    unittest.main()
