#from curses import flash
import unittest
import stratum as skrub
from stratum.logical_optimizer._optimize import optimize, OptConfig, choice_unrolling
from stratum.logical_optimizer._op_utils import show_graph, clone_sub_dag
from stratum._config import config
graph = False

class TestOpUtils(unittest.TestCase):
    def setUp(self):
        pass

    def run_clone_sub_dag(self, ops: list, clone_position: int, graph: bool = False, new_root_op = None, stop_at_op = None, run_assertions = True):
        clone_target = ops[clone_position]
        num_clone_target_children_original = len(clone_target.outputs)
        leaf = ops[-1]
        if graph:
            show_graph(ops, filename='original')
        leafs = clone_sub_dag(clone_target, new_root_op=new_root_op, stop_at_op=stop_at_op)
        if graph:
            show_graph(ops, filename='cloned')
        if run_assertions:
            # TODO Add more sophisticated expected graph comparison checks
            self.assertEqual(num_clone_target_children_original * 2, len(clone_target.outputs))


    def test_clone_sub_dag1(self):
        t1 = skrub.as_data_op(1)
        t2 = t1 + 5
        t3 = t2 - 3
        out = optimize(t3)
        self.run_clone_sub_dag(out, 0, graph=graph)

    def test_clone_sub_dag2(self):
        t1 = skrub.as_data_op(1)
        t2 = skrub.as_data_op(2.5)
        t3 = t1 / 5
        t4 = t3 + t2
        t5 = t4 - 3
        out = optimize(t5)
        self.run_clone_sub_dag(out, 0, graph=graph)

    def test_clone_sub_dag3(self):
        t1 = skrub.as_data_op(1)
        t2 = skrub.as_data_op(2.5)
        t3 = t1 + 5
        t4 = t3 + t2
        t5 = t4 - 3
        t6 = t4 * 4
        t7 = t5 + t6
        out = optimize(t7)
        self.run_clone_sub_dag(out, -4, graph=graph)

    def test_clone_sub_dag4(self):
        t1 = skrub.as_data_op(1)
        t2 = skrub.as_data_op(2.5)
        t3 = skrub.choose_from([t1, t2]).as_data_op()
        t4 = t3 + 5
        t5 = t3 - 3
        t6 = skrub.choose_from([t4, t5]).as_data_op()
        t7 = t6 + 5
        out = optimize(t7,OptConfig(cse=True, unroll_choices=False))
        self.run_clone_sub_dag(out, 2, graph=graph)

    def test_clone_sub_dag5(self):
        t1 = skrub.as_data_op(1)
        t2 = skrub.as_data_op(2.5)
        t3 = skrub.choose_from([t1, t2]).as_data_op()
        t4 = t3 + 5
        t5 = t3 - 3
        t6 = skrub.choose_from([t4, t5]).as_data_op()
        t7 = t6 + 5
        out = optimize(t7)


    def test_choice_unrolling_w_clone_sub_dag(self):
        t1 = skrub.as_data_op(1)
        t2 = skrub.as_data_op(2.5)
        t3 = skrub.choose_from([t1, t2]).as_data_op()
        t4 = t3 + 5
        t5 = t3 - 3
        t6 = skrub.choose_from([t4, t5]).as_data_op()
        t7 = t6 + 5
        out = optimize(t7, OptConfig(cse=True, unroll_choices=False))
        sink = out[-1]
        if graph:
            show_graph(sink, filename='original')
        out[1].outputs = []
        clone_sub_dag(out[2], new_root_op=out[1], stop_at_op=out[5])
        out[0].outputs = []
        for c in out[2].outputs:
            c.inputs = [out[0] if p is out[2] else p for p in c.inputs]
            out[0].outputs.append(c)
        out[2].outputs = []
        l1_names = out[2].outcome_names
        l2_names = out[5].outcome_names
        n_roots = len(l2_names)
        for i in range(n_roots):
            l2_names.append(l2_names[i] + l1_names[1])
        for i in range(n_roots):
            l2_names[i] += l1_names[0]

        if graph:
            show_graph(sink, filename='cloned')


    def test_choice_unrolling(self):
        t1 = skrub.as_data_op(1)
        t2 = skrub.as_data_op(2.5)
        t3 = skrub.choose_from([t1, t2]).as_data_op()
        t4 = t3 + 5
        t5 = t3 - 3
        t6 = skrub.choose_from([t4, t5]).as_data_op()
        t7 = t6 + 5
        out = optimize(t7, OptConfig(cse=True, unroll_choices=False))
        sink = out[-1]
        out = choice_unrolling(sink)
        with config(open_graph=False):
            show_graph(out, filename='choice_unrolling')

        


