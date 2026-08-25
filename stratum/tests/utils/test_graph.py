import pandas as pd
import unittest
import skrub
from sklearn.dummy import DummyRegressor
from stratum.utils._skrub_graph import build_graph, get_data

class TestGraph(unittest.TestCase):

    def _graph_signature(self,graph):
        """Return an id-agnostic structural summary of a skrub graph dict."""
        nodes = graph["nodes"]

        children = graph["children"]
        parents = graph["parents"]

        out_degrees = [len(children.get(node_id, [])) for node_id in nodes]
        in_degrees = [len(parents.get(node_id, [])) for node_id in nodes]

        edge_count = sum(out_degrees)

        roots = sum(1 for node_id in nodes if len(parents.get(node_id, [])) == 0)
        leaves = sum(1 for node_id in nodes if len(children.get(node_id, [])) == 0)

        in_out_pairs = sorted(zip(sorted(in_degrees), sorted(out_degrees)))

        return {
            "n_nodes": len(nodes),
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

    def test_build_graph_edges_are_deduplicated(self):
        dag = self._build_example_dag()
        fast = build_graph(dag)
        for node_id, child_ids in fast["children"].items():
            self.assertEqual(len(child_ids), len(set(child_ids)),
                             f"duplicate child edges for node {node_id}")
        for node_id, parent_ids in fast["parents"].items():
            self.assertEqual(len(parent_ids), len(set(parent_ids)),
                             f"duplicate parent edges for node {node_id}")


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

class TestGetData(unittest.TestCase):
    """`get_data` must return exactly what `dag.skb.get_data()` returns."""

    def test_matches_skrub_get_data(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        data = skrub.var("data", df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        offset = skrub.var("offset", 3)
        pred = (X + offset).skb.apply(DummyRegressor(), y=y)

        expected = pred.skb.get_data()
        actual = get_data(pred)
        # Same variables, whatever order the two traversals happen to visit them in.
        self.assertEqual(set(expected), {"data", "offset"})
        self.assertEqual(set(actual), set(expected))
        for name, value in expected.items():
            if isinstance(value, pd.DataFrame):
                pd.testing.assert_frame_equal(actual[name], value)
            else:
                self.assertEqual(actual[name], value)

    def test_skips_variables_without_a_value(self):
        # A variable declared without a value is absent from the mapping, so
        # `optimize` leaves it a VariableOp instead of folding in a bogus constant.
        valued = skrub.var("valued", 1)
        pred = valued + skrub.var("no_value")
        self.assertEqual(get_data(pred), {"valued": 1})
        self.assertEqual(get_data(pred), pred.skb.get_data())

    def test_no_variables(self):
        dag = skrub.as_data_op(pd.DataFrame({"x": [1]}))
        self.assertEqual(get_data(dag), {})
