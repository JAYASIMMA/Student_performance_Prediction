import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# === Step 1: Load the Data ===
df = pd.read_csv("score_updated.csv")  # Make sure this file is in your working directory

# === Step 2: Features and Target ===
X = df.drop(columns=["Scores"])
y = df["Scores"]

# Optional: Handle case where no categorical data exists
# Add a dummy categorical column if necessary
if X.select_dtypes(include=["object"]).empty:
    X["Category"] = "Default"

# Identify column types
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# === Step 3: Preprocessing Pipelines ===
numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# === Step 4: XGBoost Pipeline ===
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

# === Step 5: Split Data for Evaluation ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 6: Train Model ===
xgb_pipeline.fit(X_train, y_train)

# === Step 7: Evaluate ===
val_preds = xgb_pipeline.predict(X_val)
r2 = r2_score(y_val, val_preds)
print(f"✅ XGBoost R² Score on Validation Set: {r2:.4f}")

# === Step 8: Save the Model ===
joblib.dump(xgb_pipeline, "xgb_score_model.pkl")
print("📦 Model saved to 'xgb_score_model.pkl'")
