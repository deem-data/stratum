from skrub._data_ops._evaluation import _Graph
from stratum.logical_optimizer._cse import CSETable
from stratum.logical_optimizer._optimize import topological_traverse
import unittest
import stratum as skrub

class TestCSE(unittest.TestCase):
    def test_cse_table(self):
        t1 = skrub.as_data_op(1)
        t2 = skrub.as_data_op(2)
        t3 = t1 + t2
        t4 = t1 + t2

        dag = t3 + t4

        graph = _Graph().run(dag)
        nodes = graph["nodes"]
        parents = graph["parents"]
        children = graph["children"]

        order = topological_traverse(nodes, parents, children)
        table = CSETable()
        for node in order:
            table.put(node, nodes[node])
        for node in order:
            table.delete(nodes[node])
