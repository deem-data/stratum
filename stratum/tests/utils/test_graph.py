import pandas as pd
import unittest
import skrub
from sklearn.dummy import DummyRegressor
from stratum.utils._skrub_graph import build_graph

class TestGraph(unittest.TestCase):

    def _graph_signature(self,graph):
        """Return an id-agnostic structural summary of a skrub graph dict."""
        nodes = graph["nodes"]
        n = len(nodes)

        children = graph["children"]
        parents = graph["parents"]

        out_degrees = [len(children.get(i, [])) for i in range(n)]
        in_degrees = [len(parents.get(i, [])) for i in range(n)]

        edge_count = sum(out_degrees)

        roots = sum(1 for i in range(n) if len(parents.get(i, [])) == 0)
        leaves = sum(1 for i in range(n) if len(children.get(i, [])) == 0)

        in_out_pairs = sorted(zip(sorted(in_degrees), sorted(out_degrees)))

        return {
            "n_nodes": n,
            "edge_count": edge_count,
            "in_degrees_sorted": sorted(in_degrees),
            "out_degrees_sorted": sorted(out_degrees),
            "in_out_pairs": in_out_pairs,
            "n_roots": roots,
            "n_leaves": leaves,
        }


    def _build_example_dag(self):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [4, 5, 6],
            }
        )

        data = skrub.as_data_op(df)
        data_op = data.apply(lambda x: x + 1)
        X = data_op[["x"]].skb.mark_as_X()
        y = data_op["y"].skb.mark_as_y()

        pred = X.skb.apply(DummyRegressor(), y=y)
        pred = pred.skb.apply_func(lambda x, a, b: x, 1, b=1)
        pred = pred * 2
        choice = skrub.choose_from([pred], name="choice").as_data_op()

        return choice


    def test_build_graph_matches_skrub_graph_simple_dag(self):
        dag = self._build_example_dag()
        # dag.skb.draw_graph().open()
        reference = skrub._data_ops._evaluation._Graph().run(dag)
        fast = build_graph(dag)

        ref_sig = self._graph_signature(reference)
        fast_sig = self._graph_signature(fast)

        self.assertEqual(ref_sig, fast_sig)


    def test_build_graph_matches_skrub_graph_for_branching_dag(self):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [4, 5, 6],
            }
        )

        data = skrub.as_data_op(df)
        base = data.apply(lambda x: x + 1)

        # Build a slightly more complex DAG with branching.
        branch1 = base.skb.apply_func(lambda x: x * 2)
        branch2 = base.skb.apply_func(lambda x: x - 1)

        choice = skrub.choose_from([branch1, branch2], name="choice").as_data_op()

        # choice.skb.draw_graph().open()
        reference = skrub._data_ops._evaluation._Graph().run(choice)
        fast = build_graph(choice)

        ref_sig = self._graph_signature(reference)
        fast_sig = self._graph_signature(fast)

        self.assertEqual(ref_sig, fast_sig)