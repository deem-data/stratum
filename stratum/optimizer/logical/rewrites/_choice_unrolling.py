"""Expand logical choices into candidate DAGs."""
import logging

from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical._ops import ChoiceOp, Op
from stratum.utils._utils import start_time, log_time

logger = logging.getLogger(__name__)


def choice_unrolling(root: Op):
    """Unroll the choices into one candidate sub-dag per grid point.

    The result ends in a single ChoiceOp whose inputs are the candidates: the root if
    it is a choice already, otherwise one appended over the root. Choices are unrolled
    downstream-first, so the region a choice copies never holds another choice, and
    choices on different branches multiply into their Cartesian product. A candidate a
    choice does not reach is kept as is: as in skrub's grid, a choice nested in one
    outcome of another only varies when that outcome is taken.
    """
    start = start_time()
    choices = [op for op in topological_iterator(root) if op.is_choice() and op is not root]
    if choices:
        if root.is_choice():
            sink = root
        else:
            sink = ChoiceOp(outcome_names=[[]], append_choice_name=False, inputs=[root])
            root.add_output(sink)
        for choice in reversed(choices):
            logger.debug(f"Unrolling choice: {choice}")
            unroll_choice(choice, sink)
        sink.update_name()
        root = sink
    log_time("unrolled took", start)
    return root


def unroll_choice(choice: ChoiceOp, sink: ChoiceOp):
    """Replace ``choice`` by one copy of the region downstream of it per outcome.

    The region (every op between ``choice`` and ``sink``) must hold no other choice.
    The first outcome reuses the region, the others get clones. Each candidate of
    ``sink`` the region reaches becomes one candidate per outcome, named with the
    outcome's name in front of its own; the other candidates are kept as they are.
    """
    region, reached = [], {choice}
    for op in topological_iterator(sink):
        if op is not sink and any(in_ in reached for in_ in op.inputs):
            assert not op.is_choice(), "choices must be unrolled downstream-first"
            reached.add(op)
            region.append(op)

    outcomes = list(choice.inputs)
    copies = [{choice: outcome} for outcome in outcomes]
    # Clone before the region is rewired below, so clones still see the choice.
    for k in range(1, len(outcomes)):
        copy = copies[k]
        for op in region:
            clone = op.clone()
            clone.inputs = [in_ if in_ is choice else copy.get(in_, in_) for in_ in op.inputs]
            _replace_choice_input(clone, choice, outcomes[k])
            for in_ in dict.fromkeys(clone.inputs):
                in_.add_output(clone)
            copy[op] = clone
    for op in region:
        copies[0][op] = op
        if _replace_choice_input(op, choice, outcomes[0]):
            outcomes[0].add_output(op)
    choice.detach()

    names, candidates = [], []
    for copy, outcome_name in zip(copies, choice.outcome_names):
        for name, candidate in zip(sink.outcome_names, sink.inputs):
            if candidate in copy:
                names.append(outcome_name + name)
                candidates.append(copy[candidate])
            elif copy is copies[0]:
                names.append(name)
                candidates.append(candidate)
    sink.outcome_names = names
    sink.inputs = candidates
    for candidate in candidates:
        candidate.add_output(sink)


def _replace_choice_input(op: Op, choice: ChoiceOp, outcome: Op) -> bool:
    """Replace every input edge of ``op`` from ``choice`` by one from ``outcome``."""
    replaced = False
    while any(in_ is choice for in_ in op.inputs):
        op.replace_input(choice, outcome)
        replaced = True
    return replaced
