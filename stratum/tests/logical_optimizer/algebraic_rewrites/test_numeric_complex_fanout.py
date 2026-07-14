import unittest
import stratum as st
import numpy as np
from stratum.optimizer._optimize import optimize, OptConfig
from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig
from stratum.optimizer.ir._numeric_ops import NumericOp, NumericOpType
from stratum.optimizer.ir._ops import ValueOp

class TestNumericComplexFanout(unittest.TestCase):

    def _opt(self, dag):
        return optimize(dag, OptConfig(
            algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False),
        ))

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
        
        linearized_dag, *_ = self._opt(final)
        
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

        linearized_dag, *_ = self._opt(final)

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

        linearized_dag, *_ = self._opt(exp_log_a)

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
        linearized_dag, *_ = self._opt(combined)
        
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
