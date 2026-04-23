from skrub._data_ops._evaluation import _Graph
from stratum.optimizer import apply_cse_on_skrub_ir
from stratum.optimizer._cse import CSETable
from stratum.optimizer._optimize import topological_traverse
import unittest
import stratum as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# dummy function
def pre_process(df):
    return df

class TestCSE(unittest.TestCase):
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

    def test_cse_table(self):
        t1 = st.as_data_op(1)
        t2 = st.as_data_op(2)
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
        

    def test_cse(self):
        data = st.var("data", self.df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        # pipeline 1
        t1 = X.skb.apply_func(pre_process)
        y1 = t1.skb.apply(RandomForestRegressor(random_state=42), y=y)

        # pipeline 2
        t2 = X.skb.apply_func(pre_process)
        y2 = t2.skb.apply(RandomForestRegressor(random_state=123), y=y)

        y = st.choose_from({"pipeline 1": y1, "pipeline 2": y2}).as_data_op()
        y = apply_cse_on_skrub_ir(y)

    def test_cse2(self):
        data = st.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()

        # pipeline 1
        X1A = X.assign(datetime=X["datetime"].apply(lambda a: a).apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X1B = X1A.assign(
            year=X1A["datetime"].dt.year,
            month=X1A["datetime"].dt.month)
        apply_cse_on_skrub_ir(X1B)

    def test_cse4(self):
        data = st.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        # pipeline 1
        X1A = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X1B = X1A.assign(
            year=X1A["datetime"].dt.year,
            month=X1A["datetime"].dt.month,
            dayofweek=X1A["datetime"].dt.dayofweek,
            hour=X1A["datetime"].dt.hour)
        X1B = X1B.drop(["datetime"], axis=1)
        y1 = X1B.skb.apply(RandomForestRegressor(random_state=42), y=y)
        apply_cse_on_skrub_ir(y1)

    def test_cse5(self):
        data = st.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        # pipeline 1
        X1A = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X1B = X1A.assign(
            year=X1A["datetime"].dt.year,
            month=X1A["datetime"].dt.month,
            dayofweek=X1A["datetime"].dt.dayofweek,
            hour=X1A["datetime"].dt.hour)
        X1B = X1B.drop(["datetime"], axis=1)
        y1 = X1B.skb.apply(RandomForestRegressor(random_state=42), y=y)

        X2 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2B = X2.assign(
            year=X2["datetime"].dt.year,
            month=X2["datetime"].dt.month,
            dayofweek=X2["datetime"].dt.dayofweek,
            hour=X2["datetime"].dt.hour)
        X2B = X2B.drop(["datetime"], axis=1)
        y2 = X2B.skb.apply(RandomForestRegressor(random_state=123), y=y)
        y = st.choose_from({"pipeline 1": y1, "pipeline 2": y2}).as_data_op()
        apply_cse_on_skrub_ir(y)

    def test_cse6(self):
        data = st.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        # pipeline 1
        X1A = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X1B = X1A.assign(
            year=X1A["datetime"].dt.year,
            month=X1A["datetime"].dt.month,
            dayofweek=X1A["datetime"].dt.dayofweek)
        X1B = X1B.drop(["datetime"], axis=1)
        y1 = X1B.skb.apply(RandomForestRegressor(random_state=42), y=y)

        X2 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2B = X2.assign(
            month=X2["datetime"].dt.month,
            dayofweek=X2["datetime"].dt.dayofweek,
            hour=X2["datetime"].dt.hour)
        X2B = X2B.drop(["datetime"], axis=1)
        y2 = X2B.skb.apply(RandomForestRegressor(random_state=123), y=y)
        y = st.choose_from({"pipeline 1": y1, "pipeline 2": y2}).as_data_op()
        apply_cse_on_skrub_ir(y)

    def test_cse7(self):
        data = st.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()

        X1 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2A = X1.assign(
            year=X1["datetime"].dt.year,
            month=X1["datetime"].dt.month)

        apply_cse_on_skrub_ir(X2A)


    def test_cse8(self):
        data = st.var("data", self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()

        X1 = X.assign(datetime=X["datetime"].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S'))
        X2A = X1.assign(
            year=X1["datetime"].dt.year,
            month=X1["datetime"].dt.month,
            dayofweek = X1["datetime"].dt.dayofweek)
        X2B = X1.assign(
            year=X1["datetime"].dt.year,
            month=X1["datetime"].dt.month,
            dayofweek = X1["datetime"].dt.weekday)
        out = st.choose_from({"pipeline 1": X2A, "pipeline 2": X2B}).as_data_op()

        apply_cse_on_skrub_ir(out)

if __name__ == "__main__":
    unittest.main()
