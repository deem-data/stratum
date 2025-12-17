import skrub
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from stratum.logical_optimizer import apply_cse_on_skrub_ir

cols = ["Price", "Size", "Bedrooms", "Bathrooms", "Floors", "Waterfront", "View", "Condition", "Grade", "Age"]
df = pd.DataFrame({col: np.random.random(10) for col in cols})

df = skrub.as_data_op(df)
y = df["Price"].skb.mark_as_y()
x = df.drop("Price", axis=1).skb.mark_as_X()

# pipeline 1
x2 = x.assign(new_feat2= x["Bathrooms"] / x["Size"] * 100,
    new_feat3= x["Size"] * 100,
)

model = RandomForestRegressor(random_state=42, n_estimators=20, max_depth=10)
pred1 = x2.skb.apply(model, y=y)

# pipeline 2
x2 = x.assign(
    new_feat2= x["Bathrooms"] / x["Size"] * 100,
    new_feat3= x["Size"] * 100,
)
model = Ridge(random_state=42)
pred2 = x2.skb.apply(model, y=y)

preds = skrub.choose_from({"pipeline 1": pred1, "pipeline 2": pred2}).as_data_op()
preds.skb.draw_graph().open()

preds = apply_cse_on_skrub_ir(preds)
preds.skb.draw_graph().open()