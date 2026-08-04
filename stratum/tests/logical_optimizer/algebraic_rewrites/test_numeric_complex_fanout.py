import unittest
import stratum as st
import numpy as np
import scipy.special as sp
from stratum.optimizer._optimize import optimize, OptConfig
from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig
from stratum.optimizer.ir._numeric_ops import NumericOp, NumericOpType
from stratum.optimizer.ir._ops import ValueOp


def _has_softmax(dag):
    """True if any op in the plan is the fused GENERIC softmax op."""
    return any(isinstance(op, NumericOp)
               and op.type is NumericOpType.GENERIC
               and op.func is sp.softmax
               for op in dag)

class TestNumericComplexFanout(unittest.TestCase):

    def test_fanout_before_and_after_chain(self):
        """
        Scenario:
          a -> [log, d]
          log -> [exp]
          exp -> [b, c]
        
        Expected after optimization:
          a -> [d, b, c]
        """
        a = st.as_data_op(1.0)
        
        # Branch 1: log -> exp -> [b, c]
        log_a = a.skb.apply_func(np.log)
        exp_log_a = log_a.skb.apply_func(np.exp)
        b = exp_log_a + 1.0
        c = exp_log_a + 2.0
        
        # Branch 2: d
        d = a + 10.0
        
        # Root combining all branches
        t1 = d + b
        final = t1 + c
        
        linearized_dag, *_ = optimize(final)
        
        # 1. Find the original 'a' Op
        a_op = linearized_dag[0]
        self.assertIsInstance(a_op, ValueOp)
        self.assertEqual(a_op.value, 1.0)
        
        # 2. Verify 'a' has 3 outputs: d, b, and c
        self.assertEqual(len(a_op.outputs), 3, f"Expected 3 outputs for 'a', found {len(a_op.outputs)}")
        
        # All outputs of 'a' should be NumericOp(ADD) now (BinOp(+) is extracted to NumericOp)
        for out in a_op.outputs:
            self.assertIsInstance(out, NumericOp)
            self.assertEqual(out.type, NumericOpType.ADD)
            self.assertIn(a_op, out.inputs)

        # 3. Check that no log or exp ops are left
        for op in linearized_dag:
            if isinstance(op, NumericOp):
                self.assertNotIn(op.type, [NumericOpType.LOG, NumericOpType.EXP])

    def test_fanout_on_op2_only(self):
        """
        Scenario:
          a -> log -> exp -> [b, c]

        Expected after optimization:
          a -> [b, c]
        """
        a = st.as_data_op(1.0)
        log_a = a.skb.apply_func(np.log)
        exp_log_a = log_a.skb.apply_func(np.exp)
        b = exp_log_a + 1.0
        c = exp_log_a + 2.0
        final = b + c

        linearized_dag, *_ = optimize(final)

        a_op = linearized_dag[0]
        self.assertIsInstance(a_op, ValueOp)
        self.assertEqual(a_op.value, 1.0)

        self.assertEqual(len(a_op.outputs), 2)
        for out in a_op.outputs:
            self.assertIsInstance(out, NumericOp)
            self.assertEqual(out.type, NumericOpType.ADD)

        for op in linearized_dag:
            if isinstance(op, NumericOp):
                self.assertNotIn(op.type, [NumericOpType.LOG, NumericOpType.EXP])

    def test_chain_is_root(self):
        """
        Scenario:
          a -> log -> exp  (exp is the root)

        Expected after optimization:
          root is a
        """
        a = st.as_data_op(1.0)
        log_a = a.skb.apply_func(np.log)
        exp_log_a = log_a.skb.apply_func(np.exp)

        linearized_dag, *_ = optimize(exp_log_a)

        a_op = linearized_dag[0]
        self.assertIsInstance(a_op, ValueOp)
        self.assertEqual(a_op.value, 1.0)

        self.assertEqual(len(a_op.outputs), 0)
        self.assertIs(linearized_dag[-1], a_op)

        for op in linearized_dag:
            if isinstance(op, NumericOp):
                self.assertNotIn(op.type, [NumericOpType.LOG, NumericOpType.EXP])

    def test_chain_is_root_with_other_fanout(self):
        """
        Scenario:
          a -> [log, d]
          log -> exp -> BinOp (root)
        
        Expected after optimization:
          a -> [d, combined]
          d -> combined  (combined = a + d is the root)
        """
        a = st.as_data_op(1.0)
        log_a = a.skb.apply_func(np.log)
        exp_log_a = log_a.skb.apply_func(np.exp)
        
        # Add another branch so 'a' has fan-out
        d = a + 10.0
        
        combined = exp_log_a + d
        linearized_dag, *_ = optimize(combined)
        
        a_op = linearized_dag[0]
        self.assertIsInstance(a_op, ValueOp)
        self.assertEqual(a_op.value, 1.0)

        # 'a' should now connect directly to the root (the BinOp from combined)
        # and to 'd'.
        self.assertEqual(len(a_op.outputs), 2)

        # Verify no NumericOps remain
        for op in linearized_dag:
            if isinstance(op, NumericOp):
                self.assertNotIn(op.type, [NumericOpType.LOG, NumericOpType.EXP])

if __name__ == "__main__":
    unittest.main()


class TestSoftmaxFusion(unittest.TestCase):
    """`exp(x) / sum(exp(x)) -> scipy.special.softmax(x)`.

    The reduction is matched as `NumericOpType.SUM` (promoted to the enum by this
    PR) rather than by `type is GENERIC and func is np.sum`.
    """

    def test_softmax_fires(self):
        arr = np.array([1., 2., 3.])
        x = st.as_data_op(arr)
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum)

        out, *_ = optimize(e / s)
        self.assertEqual(len(out), 2)
        self.assertTrue(_has_softmax(out))
        np.testing.assert_allclose(out[1].process("fit", [arr]), sp.softmax(arr))

    def test_sum_lowers_to_typed_op(self):
        """Pin the representation the matcher depends on: np.sum must extract to
        NumericOp(SUM), not to a GENERIC op wrapping the func."""
        x = st.as_data_op(np.array([1., 2., 3.]))
        s = x.skb.apply_func(np.sum)

        out, *_ = optimize(s)
        self.assertIs(out[-1].type, NumericOpType.SUM)

    def test_sum_forwards_axis_kwarg(self):
        """SUM's process branch must forward args/kwargs, unlike the elementwise
        branches. Dropping them would turn sum(axis=0) into a whole-array sum."""
        m = np.array([[1., 2.], [3., 4.]])
        x = st.as_data_op(m)
        s = x.skb.apply_func(np.sum, axis=0)

        out, *_ = optimize(s)
        np.testing.assert_allclose(out[-1].process("fit", [m]), np.sum(m, axis=0))

    def test_sum_forwards_keepdims_kwarg(self):
        m = np.array([[1., 2.], [3., 4.]])
        x = st.as_data_op(m)
        s = x.skb.apply_func(np.sum, keepdims=True)

        out, *_ = optimize(s)
        np.testing.assert_allclose(out[-1].process("fit", [m]), np.sum(m, keepdims=True))

    def test_no_fuse_axis_aware_sum(self):
        """exp(x)/sum(exp(x), axis=0) is a column-wise softmax, not the whole-array
        one: fusing it would silently change the result."""
        m = np.array([[1., 2.], [3., 4.]])
        x = st.as_data_op(m)
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum, axis=0)

        out, *_ = optimize(e / s)
        self.assertFalse(_has_softmax(out))

    def test_no_fuse_reversed_divide(self):
        """sum(exp(x)) / exp(x) is 1/softmax(x)."""
        x = st.as_data_op(np.array([1., 2., 3.]))
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum)

        out, *_ = optimize(s / e)
        self.assertFalse(_has_softmax(out))

    def test_no_fuse_when_exp_has_third_consumer(self):
        """A third consumer of EXP still needs that value materialized."""
        x = st.as_data_op(np.array([1., 2., 3.]))
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum)

        out, *_ = optimize((e / s) + (e + 1.0))
        self.assertFalse(_has_softmax(out))

    def test_softmax_when_root(self):
        x = st.as_data_op(np.array([1., 2., 3.]))
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum)

        out, *_ = optimize(e / s)
        self.assertIs(out[-1].func, sp.softmax)

    def test_softmax_fires_mid_dag(self):
        """A consumer after the divide exercises the output-rewiring path."""
        arr = np.array([1., 2., 3.])
        x = st.as_data_op(arr)
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum)

        out, *_ = optimize((e / s) * 2.0)
        self.assertEqual(len(out), 3)
        self.assertTrue(_has_softmax(out))
        self.assertIs(out[2].type, NumericOpType.MULTIPLY)

    def test_softmax_disabled(self):
        x = st.as_data_op(np.array([1., 2., 3.]))
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(softmax=False),
        )
        out, *_ = optimize(e / s, config=config)
        self.assertFalse(_has_softmax(out))
        self.assertEqual(len(out), 4)

    def test_softmax_is_numerically_stable(self):
        """The point of the fusion: the naive form overflows, softmax does not."""
        arr = np.array([1000., 1001., 1002.])
        x = st.as_data_op(arr)
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum)

        out, *_ = optimize(e / s)
        result = out[1].process("fit", [arr])
        self.assertFalse(np.isnan(result).any())
        np.testing.assert_allclose(result, sp.softmax(arr))
