import unittest
import stratum as skrub
import numpy as np
from stratum.optimizer._optimize import  optimize
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig

class TestCSE(unittest.TestCase):

    def test_log_exp1(self):
        df = skrub.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 1)

    def test_log_exp2(self):
        df = skrub.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)
        t3 = t2.skb.apply_func(np.log1p)

        out, *_ = optimize(t3)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].value, 1)

    def test_exp_log1(self):
        df = skrub.as_data_op(1)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.log)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 1)

    def test_exp_log2(self):
        df = skrub.as_data_op(1)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.log)
        t3 = t2.skb.apply_func(np.log1p)

        out, *_ = optimize(t3)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].value, 1)

    def test_log_log1p(self):
        "no algebraic rewrite should be applied here "
        df = skrub.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.log1p)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 3)

    def test_log_log1p_exp(self):
        "no algebraic rewrite should be applied here "
        df = skrub.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.log1p)
        t3 = t2.skb.apply_func(np.exp)
        out, *_ = optimize(t3)
        self.assertEqual(len(out), 4)

    def test_log1p_log1p_exp(self):
        "no algebraic rewrite should be applied here "
        df = skrub.as_data_op(1)
        t1 = df.skb.apply_func(np.log1p)
        t2 = t1.skb.apply_func(np.log1p)
        t3 = t2.skb.apply_func(np.exp)
        out, *_ = optimize(t3)
        self.assertEqual(len(out), 4)

    def test_disable_log_exp_rewrite1(self):
        df = skrub.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(log_exp=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)

    def test_disable_log_exp_rewrite2(self):
        df = skrub.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(exp_log=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 1)

