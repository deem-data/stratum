from collections import deque
from graphviz import Digraph
from stratum.logical_optimizer._ops import Op, ChoiceOp
from ._ops import SearchEvalOp
from stratum._config import get_config

EVAL_OP_ENABLED = False


def choice_unrolling(ops_ordered: list[Op]):
    """Explode the dag after choice op into separate dags for each outcome."""
    i = 0
    while len(ops_ordered) > i:
        op = ops_ordered[i]
        if op.is_choice():
            outcomes = op.parents

            # check if we find any choice in the sub-dag of the current choice
            last_op, is_choice = find_choice_naive(op)
            no_children = last_op is op
            if no_children:
                if EVAL_OP_ENABLED:
                    # TODO add handle for no_children --> replace choice with eval op
                    raise NotImplementedError("Fix me")
                else:
                    # nothing to do
                    i += 1
                    continue
            if is_choice:
                new_ops = unroll_nested_choice(last_op, op, outcomes)
            else:
                new_ops = unroll_simple_choice(last_op, op, outcomes)

            ops_ordered.remove(op)
            ops_ordered.extend(new_ops)
            ops_ordered = topological_sort_ir(ops_ordered)
            del op
        else:
            i += 1
    return ops_ordered


def unroll_simple_choice(last_op: Op, op: ChoiceOp, outcomes: list) -> list[SearchEvalOp | ChoiceOp]:
    dag_sink = (SearchEvalOp(outcome_names=op.outcome_names, parent=last_op) if EVAL_OP_ENABLED
                          else ChoiceOp(outcome_names=op.outcome_names, append_choice_name=False))
    if not EVAL_OP_ENABLED:
        dag_sink.parents = [last_op]
        dag_sink.children = []
    new_ops = [dag_sink]

    # clones sub-dag after choice op for all outcomes[1:]
    for outcome in outcomes[1:]:
        outcome.children = []
        tmp_new_ops, leafs = clone_sub_dag(op, new_root_op=outcome)
        assert len(leafs) == 1
        dag_sink.add_parent(leafs[0])
        leafs[0].add_child(dag_sink)
        new_ops.extend(tmp_new_ops)

    # reuse sub-dag for the first outcome
    outcomes[0].children = []
    replace_op_in_children(op, replacement=outcomes[0])
    last_op.add_child(dag_sink)
    return new_ops


def unroll_nested_choice(last_op: ChoiceOp, op: ChoiceOp, outcomes) -> list[Op]:
    new_ops, n_outcomes = [], len(last_op.outcome_names)

    # clone the sub-dag for each outcome of the current choice
    for outcome, outcome_name in zip(outcomes[1:], op.outcome_names[1:]):
        outcome.children = []
        tmp_new_ops, _ = clone_sub_dag(op, new_root_op=outcome, stop_at_op=last_op)
        for i in range(n_outcomes):
            last_op.outcome_names.append(last_op.outcome_names[i] + outcome_name)
        new_ops.extend(tmp_new_ops)

    # reuse sub-dag for the first outcome
    outcomes[0].children = [op.children[0]]
    for i in range(n_outcomes):
        last_op.outcome_names[i] += op.outcome_names[0]
    outcomes[0].children = []
    replace_op_in_children(op, replacement=outcomes[0])
    return new_ops


def replace_op_in_children(op: Op, replacement: Op):
    """Replace op in children of op with replacement op."""
    for c in op.children:
        c.parents = [replacement if p is op else p for p in c.parents]
        replacement.add_child(c)


def find_choice_naive(op: Op) -> ChoiceOp:
    """
    Find the choice operation in the sub-dag using a naive approach. Might return incorrect results if there are multiple choices in the sub-dag.
    """
    # TODO check and improve find_choice(op: Op)
    last_op = op
    contains_choice = False
    while len(last_op.children) > 0 and not contains_choice:
        last_op = last_op.children[0]
        contains_choice = last_op.is_choice()
    return last_op, contains_choice


def get_all_children(op: Op, stop_at_op: Op = None):
    """Returns a list of all children. If stop_at_op is given, the children of the stop_at_op are not included."""
    queue = [op]
    visited = set()
    parents_internal = {}
    while queue:
        node = queue.pop(0)
        if node.has_children():
            for child in node.children:
                if child in visited:
                    parents_internal[child].append(node)
                elif child is not stop_at_op:
                    visited.add(child)
                    queue.append(child)
                    parents_internal[child] = [node]
                    
    return list(visited), parents_internal
    

def clone_sub_dag(root_op: Op, stop_at_op: Op = None, new_root_op: Op = None):
    """Clones a sub-dag of the given Op. Excluding the given Op, but including all its internal children. 
    Returns a list of all ops in the sub-dag, a list of the root ops of the sub-dag and a list of the leaf nodes of the sub-dag.
    """
    if new_root_op is None:
        new_root_op = root_op

    # Topological search inside sub-dag --> parents_internal contains only the parents inside the dag
    children, parents_internal = get_all_children(root_op, stop_at_op)
    indegree = {c: len(parents_internal[c]) for c in children}
    queue = []

    # clone_look_up: Look-up table for setting parents correctly
    ops_of_sub_dag, sub_dag_leaves, clone_look_up = [], [], {root_op: new_root_op}
    for c in root_op.children:
        queue.append(c)

    while queue:
        op = queue.pop(0)
        op_clone = op.clone(children=[], parents=[])
        ops_of_sub_dag.append(op_clone)
        clone_look_up[op] = op_clone

        # update op_clones's parents and the parents's chidlren
        for p in op.parents:
            p = clone_look_up.get(p, p)
            op_clone.add_parent(p)
            p.add_child(op_clone)

        if op.children is not None and len(op.children) > 0:
            for child in op.children:
                if child is stop_at_op:
                    op_clone.add_child(stop_at_op)
                    stop_at_op.add_parent(op_clone)
                    # we dont add the child to the sub-dag and dont add it to op_clone's children
                    assert len(op.children) == 1, "Op before stop Op should have only one child"
                    sub_dag_leaves.append(op_clone)
                    continue
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        else:
            sub_dag_leaves.append(op_clone)
    return ops_of_sub_dag, sub_dag_leaves


def topological_sort_ir(ops: list[Op]) -> list[Op]:
    """
    Perform topological sort on Op IR DAG.
    """
    
    indegree = {node: len(node.parents) if node.parents is not None else 0 for node in ops}
    
    # Initialize queue with nodes having no dependencies (no children)
    queue = deque([node for node, deg in indegree.items() if deg == 0])
    topo_order = []
    
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        
        # Process dependents (parents) - reduce their indegree
        if node.children is not None:
            for dependent in node.children:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
    
    # Check for cycles (if not all nodes were processed)
    if len(topo_order) != len(ops):
        for op in ops:
            if op not in topo_order:
                print(op.name)
        raise ValueError("Cycle detected in DAG - topological sort not possible")
    
    return topo_order


def show_graph(op: Op | list[Op], filename: str = 'plan'):  
    """Show the runtime plan of the DataOp DAG."""

    dot = Digraph(comment=filename)
    queue = []
    visited = set()
    if isinstance(op, list):
        for o in op:
            queue.append(o)
    else:
        queue = [op]
    while queue:
        current_op = queue.pop(0)
        if current_op in visited:
            continue
        visited.add(current_op)
        current_op.update_name()
        name = current_op.name
        name = name.replace("<","'").replace(">","'") if name is not None else "None"
        dot.node(str(id(current_op)), name)
        for child in current_op.children:
            dot.edge(str(id(current_op)), str(id(child)))
            queue.append(child)
    
    dot.render(filename, view=get_config()["open_graph"],cleanup=True)