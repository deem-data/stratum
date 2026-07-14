import unittest
import stratum as st
import numpy as np
from stratum.optimizer._optimize import  optimize
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig
from stratum.optimizer.ir._numeric_ops import NumericOp, NumericOpType
from stratum.optimizer.ir._ops import OperandRef, ValueOp
from stratum.optimizer._numeric_rewrites import match_constant_foldable

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

    def test_eliminate_identity_operation_dedups_repeated_input(self):
        """Identity-op eliminates ``*1`` multiplies; repeated inputs are properly deduplicated."""
        value = st.as_data_op(2)
        a = value + (value * 1)
        b = (value * 1) + value
        root = a + b

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(root, config=config)

        self.assertEqual(len(out), 4)
        self.assertEqual(out[1].inputs, [out[0]])
        self.assertEqual(out[2].inputs, [out[0]])
        self.assertEqual(out[2].opt_operand, OperandRef(0))
        left_sum = out[1].process("fit", [out[0].value])
        right_sum = out[2].process("fit", [out[0].value])
        self.assertEqual(out[3].process("fit", [left_sum, right_sum]), 8)

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
    
    def test_eliminate_add_zero(self):
        df = st.as_data_op(2)
        t1 = df + 0
        t2 = t1 + 3
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].process("fit", [out[0].value]), 5)

    def test_eliminate_zero_add(self):
        df = st.as_data_op(2)
        t1 = 0 + df
        t2 = t1 + 3
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].process("fit", [out[0].value]), 5)

    def test_eliminate_add_zero_root_safe(self):
        df = st.as_data_op(2)
        root = df + 0
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(root, config=config)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].process("fit", [out[0].value]), 2)

    def test_disable_eliminate_add_zero(self):
        df = st.as_data_op(2)
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(add_zero=False, constant_folding=False),
        )
        t1 = 0 + df
        t2 = t1 + 3
        out, *_ = optimize(t2, config=config)
        print(out)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[1].process("fit", [out[0].value]), 2)

    def test_no_rewrite_add_nonzero(self):
        df = st.as_data_op(2)
        t1 = df + 1
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t1, config=config)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].process("fit", [out[0].value]), 3)

    def test_add_zero_with_trailing_op(self):
        df = st.as_data_op(2)
        t1 = df + 0
        t2 = t1 + 3
        t3 = t2.skb.apply_func(np.log1p)
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t3, config=config)
        self.assertEqual(len(out), 3)

    def test_add_zero_and_identity_operation(self):
        df = st.as_data_op(2)
        t1 = df * 1
        t2 = t1 + 0
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 2)

    def test_exp_minus_one(self):
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1 - 1
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 2)                                  # source + expm1 (was 3)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.EXPM1)
        self.assertEqual(out[1].process("fit", [out[0].value]), 0)     # expm1(0) == 0

    def test_no_rewrite_one_minus_exp(self):
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.exp)
        t2 = 1 - t1                                                     # reversed: NOT expm1
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.EXP)
        self.assertEqual(out[1].process("fit", [out[0].value]), 1)

    def test_no_rewrite_exp_minus_two(self):
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1 - 2                                                     # constant != 1
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.EXP)

    def test_disable_exp_minus_one(self):
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1 - 1
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(exp_minus_one=False, constant_folding=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.EXP)

    def test_exp_minus_one_and_identity_operation(self):
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1 + 0
        t3 = t2 - 1
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t3, config=config)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.EXPM1)

    def test_log1p_of_exp_minus_one_reduces_to_input(self):
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1 - 1
        t3 = t2.skb.apply_func(np.log1p)  # log1p(exp(df)-1)
        out, *_ = optimize(t3)
        self.assertEqual(len(out), 1)  # -> log1p(expm1(df)) -> df
        self.assertEqual(out[0].value, 0)

    def test_exp_log_minus_one_not_fused(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)
        t3 = t2 - 1  # exp(log(x)) - 1
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t3, config=config)
        self.assertEqual(len(out), 2)  # exp/log cancel -> x - 1
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.SUBTRACT)

    def test_eliminate_identity_subtract(self):
        """x - 0  →  x"""
        df = st.as_data_op(5)
        t1 = df - 0
        t2 = t1 + 3

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].process("fit", [out[0].value]), 8)

    def test_eliminate_identity_subtract_root_safe(self):
        """When x - 0 is the root, the rewrite must not break the DAG."""
        value = st.as_data_op(7)
        root = value - 0

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(root, config=config)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].process("fit", [out[0].value]), 7)

    def test_disable_eliminate_identity_subtract(self):
        """Disabling identity_subtract must leave x - 0 untouched."""
        df = st.as_data_op(5)
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(identity_subtract=False, constant_folding=False),
        )
        t1 = df - 0
        t2 = t1 + 3

        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)
        subtract_result = out[1].process("fit", [out[0].value])
        self.assertEqual(subtract_result, 5)
        self.assertEqual(out[2].process("fit", [subtract_result]), 8)

    def test_no_rewrite_const_minus_var(self):
        """0 - x  should NOT be rewritten (it is not an identity)."""
        df = st.as_data_op(5)
        t1 = 0 - df
        t2 = t1 + 3

        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False))
        out, *_ = optimize(t2, config=config)
        # x - 0 would collapse to 2 ops; 0 - x stays as 3 ops
        self.assertEqual(len(out), 3)

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

    def test_constant_folding_empty_inputs_returns_none(self):
        """match_constant_foldable must return None for a NumericOp with no inputs."""
        op = NumericOp(inputs=[], outputs=[], type=NumericOpType.LOG)
        self.assertIsNone(match_constant_foldable(op))
