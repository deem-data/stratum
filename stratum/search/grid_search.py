from skrub._data_ops import DataOp
from skrub._data_ops._evaluation import _Graph
import logging
from types import SimpleNamespace
from stratum.logical_optimizer.optimize import topological_traverse
from skrub._data_ops._data_ops import Call, GetItem, CallMethod, GetAttr, Apply, Value, BinOp


logger = logging.getLogger(__name__)
def grid_search(dag: DataOp, cv: int = 5, n_jobs: int = 1):
    if logger.isEnabledFor(logging.DEBUG):
        dag.skb.draw_graph().open()

    graph = _Graph().run(dag)
    nodes = graph["nodes"]
    parents = graph["parents"]
    children = graph["children"]

    order = topological_traverse(nodes, parents, children)
    outputs = {}
    mode = "fit"

    def replace_fields_with_values(fields, children):
        child_iter = iter(children)
        
        def replace_dataop(value):
            """Recursively replace DataOp instances with their outputs."""
            if isinstance(value, DataOp):
                return outputs[next(child_iter)]
            elif isinstance(value, (list, tuple)):
                new_seq = [replace_dataop(item) for item in value]
                return type(value)(new_seq)
            elif isinstance(value, dict):
                return {key: replace_dataop(val) for key, val in value.items()}
            else:
                return value
        
        return [replace_dataop(getattr(impl, field)) for field in fields]

    # prototype sequential top down iteration of the DAG
    for node in order:
        current_op = nodes[node]
        impl = current_op._skrub_impl
        fields = impl._fields
        if isinstance(impl, Value):
            print(impl.value)
            outputs[node] = impl.value
        elif isinstance(impl, Apply):
            print(impl.estimator)
        elif isinstance(impl, Call):
            print(impl.func)
        elif isinstance(impl, CallMethod):
            print(impl.method_name)
            current_children = children[node]
            fields = replace_fields_with_values(fields, current_children)
            outputs[node] = getattr(fields[0], fields[1])(*fields[2], **fields[3])
        elif isinstance(impl, GetItem):
            print(impl.key)
            current_children = children[node]
            container = outputs[current_children[0]]
            outputs[node] = container[impl.key]
        elif isinstance(impl, GetAttr):
            current_children = children[node]
            container = outputs[current_children[0]]
            outputs[node] = getattr(container, impl.attr_name)
            print(impl.attr_name)
        elif isinstance(impl, BinOp):
            print(impl.op)
        else:
            print(impl)