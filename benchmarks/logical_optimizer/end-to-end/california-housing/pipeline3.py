import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet

# Load dataset
df = pd.read_csv("input/train.csv")
target = "MedHouseVal"

# Feature engineering
def feat_eng(X):
    return X.assign(
        BedroomsPerRoom=X["AveBedrms"] / X["AveRooms"],
        IncomeSquared=X["MedInc"] ** 2,
        IncomeRoomInteraction=X["MedInc"] * X["AveRooms"],
        Density=X["Population"] / X["AveOccup"],
        LatitudeLongitude=X["Latitude"] * X["Longitude"],
        MedInc3=X["MedInc"] ** 3,
        RoomDensity=X["AveRooms"] / X["Population"]
    )

# Prepare features and target
X = df.drop(columns=[target])
X = feat_eng(X)
y = df[target]

numeric_features = X.columns.tolist()

# Build preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features)
    ]
)

# Build model pipeline
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", ElasticNet())
])

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
mse_scores = cv_scores.mean()

print(f"Pipeline 3 (ElasticNet) MSE: {mse_scores:.4f}")

