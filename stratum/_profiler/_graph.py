"""Graphviz plan rendering for stratum-gui snapshots (same layout as show_graph)."""
from __future__ import annotations

from graphviz import Digraph

from stratum._config import get_config
from stratum.optimizer._op_utils import validate_operands
from stratum.optimizer.logical._ops import ChoiceOp


def plan_graph(nodes, filename: str = 'plan'):
    """Graph of snapshot nodes ({"id", "label", "inputs"}), with stable SVG node IDs."""
    dot = Digraph(
        comment=filename,
        format=get_config()["graph_format"],
        graph_attr={'rankdir': 'BT', 'ranksep': '0.3', 'nodesep': '0.2'},
        node_attr={'fontname': 'Helvetica', 'fontsize': '11'},
    )
    for node in nodes:
        label = node["label"]
        label = label.replace("<", "'").replace(">", "'") if label is not None else "None"
        dot.node(str(node["id"]), label, id=f"node-{node['id']}")
        for input_id in node["inputs"]:
            dot.edge(str(input_id), str(node["id"]))
    return dot


def op_graph(ops, filename: str = 'plan'):
    """Plan graph with stage-local IDs for node inspection."""
    ops = list(ops)
    ids = {op: i for i, op in enumerate(ops)}
    nodes = []
    for op in ops:
        validate_operands(op)
        op.update_name()
        nodes.append({"id": ids[op],
                      "label": op.name if isinstance(op, ChoiceOp) else str(op),
                      "inputs": [ids[inp] for inp in op.inputs]})
    return plan_graph(nodes, filename)


def graph_svg(dot):
    """Render in memory, excluding the XML declaration and external DTD."""
    svg = dot.pipe(format="svg").decode("utf-8")
    return svg[svg.index("<svg"):]
