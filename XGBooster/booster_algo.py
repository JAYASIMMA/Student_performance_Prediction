import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

df = pd.read_csv("score_updated.csv") 

X = df.drop(columns=["Scores"])
y = df["Scores"]

if X.select_dtypes(include=["object"]).empty:
    X["Category"] = "Default"

numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_pipeline.fit(X_train, y_train)

val_preds = xgb_pipeline.predict(X_val)
r2 = r2_score(y_val, val_preds)
print(f"✅ XGBoost R² Score on Validation Set: {r2:.4f}")

joblib.dump(xgb_pipeline, "xgb_score_model.pkl")
print("📦 Model saved to 'xgb_score_model.pkl'")
