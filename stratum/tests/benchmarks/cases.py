"""Predefined benchmark cases for the Stratum optimizer.

Each case is a :class:`BenchmarkCase` describing a DataOp DAG, the rewrite
rules it should exercise, and a brief description of what is being tested.

To add a new case, create a ``BenchmarkCase`` and append it to ``ALL_CASES``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from skrub import DataOp

import stratum as st


@dataclass
class BenchmarkCase:
    """A single benchmark case.

    Attributes
    ----------
    name : str
        Short identifier (e.g. ``"log_exp_inverse"``).
    description : str
        Human-readable explanation.
    dag_builder : Callable[[pd.DataFrame], DataOp]
        Builds the DataOp DAG from a DataFrame.  Receives a fresh DataFrame
        each invocation so cases are isolated.
    expected_rules : set[str]
        Names of rewrite rules this case is expected to trigger (from
        ``AlgebraicRewritesConfig`` field names).
    category : str
        Grouping label (``"numeric"``, ``"dataframe"``, ``"combined"``).
    """

    name: str
    description: str
    dag_builder: Callable[..., DataOp]
    expected_rules: set[str] = field(default_factory=set)
    category: str = "numeric"


# ---------------------------------------------------------------------------
# Numeric algebraic rewrite cases
# ---------------------------------------------------------------------------

NUMERIC_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        name="log_exp_inverse",
        description="log(exp(x)) → x  —  two ops eliminated",
        dag_builder=lambda df: st.as_data_op(df)
        .skb.apply_func(np.log)
        .skb.apply_func(np.exp),
        expected_rules={"log_exp"},
        category="numeric",
    ),
    BenchmarkCase(
        name="exp_log_inverse",
        description="exp(log(x)) → x  —  two ops eliminated",
        dag_builder=lambda df: st.as_data_op(df)
        .skb.apply_func(np.exp)
        .skb.apply_func(np.log),
        expected_rules={"exp_log"},
        category="numeric",
    ),
    BenchmarkCase(
        name="log1p_expm1_inverse",
        description="log1p(expm1(x)) → x",
        dag_builder=lambda df: st.as_data_op(df)
        .skb.apply_func(np.log1p)
        .skb.apply_func(np.expm1),
        expected_rules={"log1p_expm1"},
        category="numeric",
    ),
    BenchmarkCase(
        name="expm1_log1p_inverse",
        description="expm1(log1p(x)) → x",
        dag_builder=lambda df: st.as_data_op(df)
        .skb.apply_func(np.expm1)
        .skb.apply_func(np.log1p),
        expected_rules={"expm1_log1p"},
        category="numeric",
    ),
    BenchmarkCase(
        name="sqrt_square_to_abs",
        description="sqrt(x²) → |x|  —  replaced by abs",
        dag_builder=lambda df: st.as_data_op(df)
        .skb.apply_func(np.square)
        .skb.apply_func(np.sqrt),
        expected_rules={"sqrt_square"},
        category="numeric",
    ),
    BenchmarkCase(
        name="multiply_by_one",
        description="x * 1 → x  —  identity multiply eliminated",
        dag_builder=lambda df: st.as_data_op(df) * 1,
        expected_rules={"identity_op"},
        category="numeric",
    ),
    BenchmarkCase(
        name="add_zero",
        description="x + 0 → x  —  identity add eliminated",
        dag_builder=lambda df: st.as_data_op(df) + 0,
        expected_rules={"add_zero"},
        category="numeric",
    ),
    BenchmarkCase(
        name="subtract_zero",
        description="x - 0 → x  —  identity subtract eliminated",
        dag_builder=lambda df: st.as_data_op(df) - 0,
        expected_rules={"identity_subtract"},
        category="numeric",
    ),
    BenchmarkCase(
        name="abs_abs_collapse",
        description="abs(abs(x)) → abs(x)",
        dag_builder=lambda df: st.as_data_op(df)
        .skb.apply_func(np.abs)
        .skb.apply_func(np.abs),
        expected_rules={"abs_abs"},
        category="numeric",
    ),
    BenchmarkCase(
        name="exp_minus_one_to_expm1",
        description="exp(x) - 1 → expm1(x)",
        dag_builder=lambda df: st.as_data_op(df).skb.apply_func(np.exp) - 1,
        expected_rules={"exp_minus_one"},
        category="numeric",
    ),
    BenchmarkCase(
        name="multiply_by_zero",
        description="x * 0 → ValueOp(0.0)  —  dead subgraph pruned",
        dag_builder=lambda df: st.as_data_op(df) * 0,
        expected_rules={"any_mul_zero"},
        category="numeric",
    ),
]

# ---------------------------------------------------------------------------
# Combined / multi-rewrite cases
# ---------------------------------------------------------------------------

COMBINED_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        name="combined_log_exp_identity",
        description="log(exp(x))*1 + 0 → x  —  log_exp + identity + add_zero cascade",
        dag_builder=lambda df: (
            st.as_data_op(df).skb.apply_func(np.log).skb.apply_func(np.exp) * 1 + 0
        ),
        expected_rules={"log_exp", "identity_op", "add_zero"},
        category="combined",
    ),
    BenchmarkCase(
        name="combined_sqrt_square_with_identity",
        description="sqrt(x²)*1 - 0 → |x|  —  sqrt_square + identity + subtract_zero",
        dag_builder=lambda df: (
            st.as_data_op(df).skb.apply_func(np.square).skb.apply_func(np.sqrt) * 1 - 0
        ),
        expected_rules={"sqrt_square", "identity_op", "identity_subtract"},
        category="combined",
    ),
    BenchmarkCase(
        name="combined_many_rewrites",
        description="log(exp(x))*1+0-0 → sqrt(square(x)) → abs(x)  —  5+ rewrites",
        dag_builder=lambda df: (
            st.as_data_op(df)
            .skb.apply_func(np.log).skb.apply_func(np.exp)  # log_exp
            * 1  # identity
            + 0  # add_zero
            - 0  # identity_subtract
        ),
        expected_rules={"log_exp", "identity_op", "add_zero", "identity_subtract"},
        category="combined",
    ),
    BenchmarkCase(
        name="combined_full_pipeline",
        description="log→exp→*1→+0→square→sqrt→exp→-1  —  most numeric rewrites",
        dag_builder=lambda df: (
            st.as_data_op(df)
            .skb.apply_func(np.log).skb.apply_func(np.exp)  # log_exp
            * 1  # identity
            + 0  # add_zero
        ).skb.apply_func(np.square).skb.apply_func(np.sqrt),  # sqrt_square
        expected_rules={"log_exp", "identity_op", "add_zero", "sqrt_square"},
        category="combined",
    ),
]

# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------

ALL_CASES: list[BenchmarkCase] = NUMERIC_CASES + COMBINED_CASES
