import unittest
import stratum as st
import numpy as np
from stratum.optimizer._optimize import  optimize
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig
from stratum.optimizer.ir._numeric_ops import NumericOp, NumericOpType

class TestCSE(unittest.TestCase):

    def test_log_exp1(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 1)

    def test_log_exp2(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)
        t3 = t2.skb.apply_func(np.log1p)

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t3, config=config)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].value, 1)

    def test_exp_log1(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.log)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 1)

    def test_exp_log2(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.log)
        t3 = t2.skb.apply_func(np.log1p)

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t3, config=config)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].value, 1)

    def test_log1p_expm1(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log1p)
        t2 = t1.skb.apply_func(np.expm1)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 1)

    def test_expm1_log1p(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.expm1)
        t2 = t1.skb.apply_func(np.log1p)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 1)

    def test_log_log1p(self):
        "no algebraic rewrite should be applied here "
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.log1p)

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)

    def test_expm1_log1p_disabled(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.expm1)
        t2 = t1.skb.apply_func(np.log1p)
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(log1p_expm1 = False,expm1_log1p = False, constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].value, 1)

    def test_log_log1p_exp(self):
        "no algebraic rewrite should be applied here "
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.log1p)
        t3 = t2.skb.apply_func(np.exp)
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t3, config=config)
        self.assertEqual(len(out), 4)

    def test_log1p_log1p_exp(self):
        "no algebraic rewrite should be applied here "
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log1p)
        t2 = t1.skb.apply_func(np.log1p)
        t3 = t2.skb.apply_func(np.exp)
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t3, config=config)
        self.assertEqual(len(out), 4)

    def test_disable_log_exp_rewrite1(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(log_exp=False, constant_folding=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)

    def test_disable_log_exp_rewrite2(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(exp_log=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 1)

    def test_sqrt_square_via_np_square(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.square)
        t2 = t1.skb.apply_func(np.sqrt)

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 2)
        # abs(4) = 4
        self.assertEqual(out[0].value, 4)

    def test_sqrt_pow2(self):
        df = st.as_data_op(-3)
        t1 = df ** 2
        t2 = t1.skb.apply_func(np.sqrt)

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 2)
        # abs(-3) = 3
        self.assertEqual(out[0].value, -3)

    def test_sqrt_square_with_trailing_op(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.square)
        t2 = t1.skb.apply_func(np.sqrt)
        t3 = t2.skb.apply_func(np.log1p)

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t3, config=config)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].value, 4)

    def test_disable_sqrt_square_rewrite(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.square)
        t2 = t1.skb.apply_func(np.sqrt)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(sqrt_square=False, constant_folding=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)

    def test_no_rewrite_sqrt_only(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.sqrt)

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t1, config=config)
        self.assertEqual(len(out), 2)

    def test_sqrt_square_produces_abs_op(self):
        """Rewrite must insert an abs op, not identity — critical for negative inputs."""
        df = st.as_data_op(-5)
        t1 = df.skb.apply_func(np.square)
        t2 = t1.skb.apply_func(np.sqrt)

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.ABS)

    def test_sqrt_pow2_produces_abs_op(self):
        """BinOp(**2) → sqrt rewrite must also produce abs, not identity."""
        df = st.as_data_op(-5)
        t1 = df ** 2
        t2 = t1.skb.apply_func(np.sqrt)

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.ABS)

    def test_no_rewrite_square_log(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.square)
        t2 = t1.skb.apply_func(np.log)

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)

    def test_no_rewrite_sqrt_exp(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.sqrt)
        t2 = t1.skb.apply_func(np.exp)

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)

    def test_no_rewrite_pow3_sqrt(self):
        """x**3 → sqrt should not rewrite; only x**2 qualifies."""
        df = st.as_data_op(4)
        t1 = df ** 3
        t2 = t1.skb.apply_func(np.sqrt)

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)

    def test_disable_sqrt_square_does_not_affect_log_exp(self):
        """Disabling sqrt_square must not suppress other algebraic rewrites."""
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(sqrt_square=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 1)

    def test_eliminate_identity_operation(self):
        df = st.as_data_op(2)
        t1 = df * 1
        t2 = t1 + 3

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].process("fit", [out[0].value]), 5)

    def test_disable_eliminate_identity_operation(self):
        df = st.as_data_op(2)
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(identity_op=False, constant_folding=False),
        )
        t1 = df * 1
        t2 = t1 + 3

        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)
        multiply_result = out[1].process("fit", [out[0].value])
        self.assertEqual(multiply_result, 2)
        self.assertEqual(out[2].process("fit", [multiply_result]), 5)

    def test_eliminate_identity_operation_root_safe(self):
        value = st.as_data_op(2)
        root = value * 1

        out, *_ = optimize(root)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].process("fit", [out[0].value]), 2)

    def test_abs_abs_collapses_to_single_abs(self):
        df = st.as_data_op(-3)
        t1 = df.skb.apply_func(np.abs)
        t2 = t1.skb.apply_func(np.abs)
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)

        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.ABS)

    def test_single_abs_untouched(self):
        df = st.as_data_op(-3)
        t1 = df.skb.apply_func(np.abs)
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t1, config=config)
        self.assertEqual(len(out), 2)

    def test_abs_abs_with_trailing_op(self):
        df = st.as_data_op(-3)
        t1 = df.skb.apply_func(np.abs)
        t2 = t1.skb.apply_func(np.abs)
        t3 = t2.skb.apply_func(np.log1p)
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t3, config=config)
        self.assertEqual(len(out), 3)

    def test_no_rewrite_abs_sqrt(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.abs)
        t2 = t1.skb.apply_func(np.sqrt)
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)

    def test_abs_abs_disabled(self):
        df = st.as_data_op(-3)
        t1 = df.skb.apply_func(np.abs)
        t2 = t1.skb.apply_func(np.abs)
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(abs_abs=False, constant_folding=False)
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)

    def test_disable_abs_abs_does_not_affect_log_exp(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(abs_abs=False)
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 1)

    # --- constant folding -------------------------------------------------

    def test_constant_folding_log1(self):
        """log(1) → 0"""
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 0.0)

    def test_constant_folding_exp0(self):
        """exp(0) → 1"""
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.exp)

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 1.0)

    def test_constant_folding_abs(self):
        """abs(-5) → 5"""
        df = st.as_data_op(-5)
        t1 = df.skb.apply_func(np.abs)

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 5)

    def test_constant_folding_sqrt(self):
        """sqrt(4) → 2"""
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.sqrt)

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 2.0)

    def test_constant_folding_square(self):
        """square(3) → 9"""
        df = st.as_data_op(3)
        t1 = df.skb.apply_func(np.square)

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 9)

    def test_constant_folding_log1p(self):
        """log1p(e-1) → 1"""
        df = st.as_data_op(np.e - 1)
        t1 = df.skb.apply_func(np.log1p)

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 1)
        np.testing.assert_almost_equal(out[0].value, 1.0)

    def test_constant_folding_expm1(self):
        """expm1(1) → e-1"""
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.expm1)

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, np.expm1(1))

    def test_constant_folding_sin0(self):
        """sin(0) → 0 (GENERIC numeric op)"""
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.sin)

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 0.0)

    def test_constant_folding_cos0(self):
        """cos(0) → 1 (GENERIC numeric op)"""
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.cos)

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 1.0)

    def test_constant_folding_add_var_var(self):
        """5 + 3 → 8 (binary var-var)"""
        df1 = st.as_data_op(5)
        df2 = st.as_data_op(3)
        t1 = df1 + df2

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 8)

    def test_constant_folding_add_var_const(self):
        """5 + 3 → 8 (binary var-const)"""
        df = st.as_data_op(5)
        t1 = df + 3

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 8)

    def test_constant_folding_cascade(self):
        """sqrt(abs(-4)) → 2 (two-level cascade)"""
        df = st.as_data_op(-4)
        t1 = df.skb.apply_func(np.abs)
        t2 = t1.skb.apply_func(np.sqrt)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 2.0)

    def test_constant_folding_disabled(self):
        """Disabling constant_folding must leave log(1) as a NumericOp."""
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False),
        )
        out, *_ = optimize(t1, config=config)
        self.assertEqual(len(out), 2)  # ValueOp + NumericOp

    def test_disable_constant_folding_does_not_affect_log_exp(self):
        """Disabling constant_folding must not suppress other rewrites."""
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 1)  # log(exp(x)) eliminated
        self.assertEqual(out[0].value, 1)

