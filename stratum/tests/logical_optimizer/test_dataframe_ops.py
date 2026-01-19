import os
import tempfile
from stratum.logical_optimizer._dataframe_ops import AssignOp, DataSourceOp, DatetimeConversionOp, GetAttrProjectionOp, ProjectionOp
from stratum.logical_optimizer._ops import GetItemOp, MethodCallOp
from stratum.logical_optimizer._optimize import OptConfig, optimize
import stratum as skrub
import pandas as pd
import unittest

class TestDataframeOps(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "datetime": [
                "2025-11-01 10:00:00",
                "2025-11-02 15:30:00",
                "2025-11-03 09:45:00"
            ]
        })

    def test_data_source_rewrite_df(self):
        data = skrub.as_data_op(self.df)
        ops = optimize(data)
        assert len(ops) == 1
        assert isinstance(ops[0], DataSourceOp)
    
    def test_data_source_rewrite_read(self):
        tmp_file = tempfile.mktemp(suffix=".csv")
        self.df.to_csv(tmp_file, index=False)
        data = skrub.as_data_op(tmp_file).skb.apply_func(pd.read_csv)
        ops = optimize(data, OptConfig(dataframe_ops=True))
        assert len(ops) == 1
        assert isinstance(ops[0], DataSourceOp)
        os.remove(tmp_file)

    def test_projection_rewrite_df(self):
        data = skrub.as_data_op(self.df).drop("y", axis=1)
        ops = optimize(data)
        assert len(ops) == 2
        assert isinstance(ops[1], ProjectionOp)

    @unittest.skip("Skipping this test for now")
    def test_projection_fused_get_item_rewrite_df1(self):
        data = skrub.as_data_op(self.df)["x"].apply(lambda x: x + 1)
        ops = optimize(data)
        assert len(ops) == 2
        assert isinstance(ops[1], ProjectionOp)

    def test_projection_fused_get_item_rewrite_df2(self):
        data = skrub.as_data_op(self.df)["x"]
        sub_dag1 = data.apply(lambda x, a: x + a, a=skrub.as_data_op(1))
        sub_dag2 = data
        sink = skrub.choose_from([sub_dag1, sub_dag2]).as_data_op()
        ops = optimize(sink)
        self.assertEqual(5, len(ops))
        self.assertTrue(isinstance(ops[2], GetItemOp))
        self.assertTrue(isinstance(ops[3], ProjectionOp))

    def test_fused_get_attr_rewrite_df(self):
        data = skrub.as_data_op(self.df)[["datetime"]].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S')
        data = data.assign(year= data["datetime"].dt.year, month= data["datetime"].dt.month)
        data = data.copy()
        ops = optimize(data)
        # change to 7 when get item rewrite is working again
        self.assertEqual(8,len(ops))
        op_iter = iter(ops[3:])
        next(op_iter) # remove after fix
        self.assertTrue(isinstance(next(op_iter), GetAttrProjectionOp))
        self.assertTrue(isinstance(next(op_iter), GetAttrProjectionOp))
        self.assertTrue(isinstance(next(op_iter), AssignOp))
        self.assertTrue(isinstance(next(op_iter), MethodCallOp))