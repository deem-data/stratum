from skrub._data_ops import DataOp
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .logical.rewrites._common_subexpression_elimination import apply_op_cse
from .logical.rewrites._candidate_construction import install_candidate_set
from .logical.rewrites._choice_unrolling import choice_unrolling
from .logical.rewrites._schema_propagation import propagate_output_schema
from .logical._dataframe_ops import extract_dataframe_op, add_splitting_op
from .logical.rewrites._literal_inlining import literal_inlining
from .logical._numeric_ops import extract_numeric_op
from .logical._ops import Op
from ._op_utils import show_graph, topological_iterator, validate_dag
from ._explain import explain_linear_plan
from .logical.rewrites._algebraic_rewrites import algebraic_rewrites, AlgebraicRewritesConfig
from .logical.rewrites._relational_rewrites import relational_rewrites
from .physical.rewrites._linearization import linearize_dag
from .physical.rewrites._dead_code_elimination import mark_dead_ops_during_fit
from .physical.rewrites._lifetime_planning import compute_pinned_ops, plan_input_removals
from .physical.rewrites._reuse_planning import apply_reuse_cache
from .physical._plan_context import PlanContext
from .physical._lowering import lower_to_physical
from .physical._impl_selection import (ImplementationSelector, get_implementation_selector, select_implementations)
from stratum.frontend._conversion import convert_to_ops
from stratum._config import FLAGS
from stratum._profiler._diagnostics import capture_plan, diagnostics_enabled
from stratum.utils._utils import start_time, log_time

if TYPE_CHECKING:
    from stratum.runtime._reuse_cache import ReuseCache


@dataclass(frozen=True)
class SearchConfig:
    """What a plan needs at build time to score the candidates it evaluates.

    Present only when the plan is being built for a search; ``evaluate`` passes None and
    gets a plan that collects candidates without scoring them.
    """

    metric: Any
    return_predictions: bool = False
    #: The metric is the one the plan declares with `.skb.with_scoring()`, so the
    #: scorer also reads the kwargs declared with it, computed on each test fold.
    declared_kwargs: bool = False


class OptConfig():
    def __init__(
        self,
        cse: bool = True,
        unroll_choices: bool = True,
        dataframe_ops: bool = True,
        numeric_ops: bool = True,
        algebraic_rewrites: bool = True,
        algebraic_rewrite_config: AlgebraicRewritesConfig | None = None,
        propagate_schema: bool = True,
        semi_join_rewrite: bool = True,
        projection_pushup: bool = True,
        selector: ImplementationSelector | None = None,
        pandas_query: bool = False,
        literal_inlining: bool = True,
    ):
        """Configure optimization and implementation selection for one plan.

        - ``selector=None`` uses ``FLAGS.implementation_selector``. An explicit
          selector can select a policy without a public config name, such as
          ``FlagBasedSelector(backend="polars")`` for testing Polars implementations.
        - ``pandas_query`` enables ``PandasQuerySelectionOp`` as a candidate for
          masks whose predicates compile to query strings. This optimizer-only
          setting selects between pandas kernels; ``set_config`` does not expose
          it, and ``make_grid_search`` uses the default.
        """
        self.cse = cse
        self.dataframe_ops = dataframe_ops
        self.unroll_choices = unroll_choices
        self.numeric_ops = numeric_ops
        self.algebraic_rewrites = algebraic_rewrites
        self.semi_join_rewrite = semi_join_rewrite
        self.projection_pushup = projection_pushup
        if algebraic_rewrite_config is None:
            algebraic_rewrite_config = AlgebraicRewritesConfig()
        self.algebraic_rewrite_config = algebraic_rewrite_config
        self.propagate_schema = propagate_schema
        self.selector = selector
        self.pandas_query = pandas_query
        self.literal_inlining = literal_inlining

def _debug_show_graph(root: Op, name: str):
    if diagnostics_enabled():
        capture_plan("selected_physical" if name == "physical" else name,
                     topological_iterator(root))
    if FLAGS.debug_graph:
        show_graph(root, name)

def _debug_explain_linear_plan(name: str, linearized_dag: list, split_pos: int | None):
    """Print the final executable plan (post implementation selection)."""
    if "physical_impl" in FLAGS.explain:
        explain_linear_plan(name, linearized_dag, split_pos)


def _debug_explain_dag(level: str, root: Op):
    """Print a topological plan for a non-linearized IR level (logical/physical).

    These levels have not been linearized around the split operator, so display
    a flat topological listing (``split_pos=None``)."""
    if diagnostics_enabled():
        capture_plan(level, topological_iterator(root))
    if level in FLAGS.explain:
        explain_linear_plan(level, list(topological_iterator(root)), split_pos=None)


def _debug_validate_dag(root: Op):
    """Assert every OperandRef indexes a valid input edge across the whole DAG.

    With FLAGS.validate_dag enabled, reject out-of-range operand references
    left by rewrites that remove or reorder inputs."""
    if FLAGS.validate_dag:
        validate_dag(root)

def optimize(dag_root: DataOp, config: OptConfig = None, env: dict = None,
             search: SearchConfig | None = None, reuse_cache: "ReuseCache | None" = None):
    """Entry point for the optimizer. Runs the three planning phases and returns
    the linearized physical plan ``(linearized_dag, split_pos, flagged_ops)``.

    The steps are:

    1. :func:`logical_optimize` -- compile the Skrub DataOp DAG to the logical
       IR and run all backend-agnostic rewrites (extraction, CSE, choice
       unrolling, algebraic rewrites).
    2. :func:`~stratum.optimizer.physical._lowering.lower_to_physical` -- lower
       logical ops to physical ops (one logical op may become several).
    3. :func:`physical_optimize` -- select a concrete implementation per
       physical op, optionally prune work found in ``reuse_cache``, then linearize
       and plan intermediate lifetimes.

    ``env`` (variable name -> value), when supplied, lets the converter resolve
    variables to compile-time constants (ValueOps) instead of VariableOps. ``search``,
    when supplied, makes this a search plan: it ends in a scoring operator rather than
    one that only labels its candidates. ``reuse_cache`` enables planning against
    an explicit in-memory cache; cache hits require a scheduler that consumes them."""
    start = start_time()
    if config is None:
        config = OptConfig()

    # Step 1: Convert the Skrub DataOp DAG to the logical IR and apply rewrites.
    root = logical_optimize(dag_root, config, env, search)
    _debug_explain_dag("logical", root)

    # Steps 2 & 3 read the config that drives operator selection once, here, so
    # execution carries no operator-selection control flow.
    ctx = PlanContext.from_flags(config)

    # Step 2: lower logical ops to physical ops.
    root = lower_to_physical(root, ctx)
    _debug_validate_dag(root)  # operand refs after lowering
    _debug_show_graph(root, "lowered")
    _debug_explain_dag("physical", root)

    # Step 3: physical optimization (implementation selection + linearization).
    # TODO: May need physical-level rewrites before or after operator selection
    result = physical_optimize(root, ctx, selector=config.selector, reuse_cache=reuse_cache)

    log_time("Optimization took in total", start)
    return result


def logical_optimize(dag_root: DataOp, config: OptConfig, env: dict = None,
                     search: SearchConfig | None = None) -> Op:
    """Step 1: build the logical IR and run all backend-agnostic rewrites.
    Returns the logical DAG root, ready to be lowered to physical ops."""

    # Convert to logical operator DAG
    root = convert_to_ops(dag_root, env)
    _debug_show_graph(root, "conversion")
    if config.literal_inlining:
        root = literal_inlining(root)

    # Add splitting op
    root = add_splitting_op(root)
    _debug_validate_dag(root)  # operand refs as wired by as_op

    # Extract specialized operators from generic MethodCallOp / CallOp.
    if config.dataframe_ops:
        root = extract_frame_operators(root)
    if config.numeric_ops:
        root = extract_numeric_operators(root)

    # Apply CSE on the Op IR *after* extraction, so it can dedup whole specialized
    # ops (e.g. two identical mask SelectionOps). Running it earlier would merge a
    # mask's shared sub-expressions into a node with multiple consumers, which then
    # blocks selection folding.
    if FLAGS.cse:
        root = run_op_cse_pass(root)

    # Unrolling of choices to a dag with only a single ChoiceOp at the end
    if config.unroll_choices:
        root = choice_unrolling(root)
        _debug_show_graph(root, "unrolled")

    # Final logical DAG
    if config.algebraic_rewrites:
        root = algebraic_rewrites(root, config.algebraic_rewrite_config)
        _debug_show_graph(root, "algebraic_rewrite")

    # Projection pushup reads schemas, so propagate them before relational rewrites.
    # These rewrites must maintain schemas for the operators they create or change.
    if config.propagate_schema:
        propagate_output_schema(root)

    # Run relational rewrites after expression rewrites and CSE so a converted
    # join can reuse the build-side operator shared by other consumers.
    root = relational_rewrites(root, semi_join=config.semi_join_rewrite,
                               projection_pushup=config.projection_pushup)
    _debug_show_graph(root, "relational_rewrite")

    # Last, so every rewrite above sees the plan shape it was written against. A plan
    # whose choices were not unrolled is not an executable candidate set, so it gets no
    # terminating operator.
    if config.unroll_choices:
        root = install_candidate_set(root, search)

    # Once more over the final shape, for the ops created since that do not keep
    # their own schema (the candidate set). Under validate_dag it also checks that
    # the relational rewrites left every schema they had current.
    if config.propagate_schema:
        propagate_output_schema(root, check=FLAGS.validate_dag)

    _debug_validate_dag(root)  # operand refs after all logical rewrites, before lowering
    _debug_show_graph(root, "final_logical_plan")
    return root


def physical_optimize(root: Op, ctx: PlanContext, registry=None,
                      selector: ImplementationSelector | None = None,
                      reuse_cache: "ReuseCache | None" = None):
    """Step 3: select implementations, plan reuse, then linearize and plan lifetimes.
    Implementation selection resolves each op with registered candidates to a
    concrete implementation (consulting the default PhysicalRegistry unless one
    is injected). An explicit cache supplies hits and admission keys before
    linearization, so lifetime planning and dead code elimination see the
    pruned graph."""

    # TODO: add physical rewrites here (or afterwards)
    if selector is None:
        selector = get_implementation_selector(ctx.implementation_selector)
    root = select_implementations(root, ctx, registry=registry, selector=selector)
    _debug_validate_dag(root)
    if reuse_cache is not None:
        apply_reuse_cache(root, reuse_cache)
        _debug_validate_dag(root)
    _debug_show_graph(root, "physical_impl")
    # Parallel execution planning could also run here, before linearization.

    # Linearize and plan intermediate retention and releases.
    linearized_dag, split_pos, flagged_ops = linearize_dag(root)
    pinned_ops = compute_pinned_ops(linearized_dag, split_pos, flagged_ops)
    plan_input_removals(linearized_dag, pinned_ops)
    mark_dead_ops_during_fit(linearized_dag, split_pos, flagged_ops)

    if diagnostics_enabled():
        capture_plan("physical_impl", linearized_dag, split_pos, flagged_ops)
    _debug_explain_linear_plan("physical_impl", linearized_dag, split_pos)
    return linearized_dag, split_pos, flagged_ops


def run_op_cse_pass(root: Op) -> Op:
    """Apply CSE on the Op IR (post-conversion) and return the deduplicated root."""
    start = start_time()
    root = apply_op_cse(root)
    log_time("Op CSE took", start)
    _debug_validate_dag(root)
    _debug_show_graph(root, "op_cse")
    return root


def extract_frame_operators(root):
    """Extract specialized dataframe operators from generic calls in the DAG."""
    start = start_time()
    for op in topological_iterator(root):
        root, _ = extract_dataframe_op(op, root, FLAGS.make_selection_op,
                                       FLAGS.make_map_op, FLAGS.make_column_projection)
    log_time("dataframe_rewrite took", start)
    _debug_show_graph(root, "frame_rewrite")
    return root


def extract_numeric_operators(root):
    """Extract numeric operators from generic calls in the DAG."""
    start = start_time()
    for op in topological_iterator(root):
        root, _ = extract_numeric_op(op, root)
    log_time("to_numeric took", start)
    _debug_show_graph(root, "numeric_rewrite")
    return root
