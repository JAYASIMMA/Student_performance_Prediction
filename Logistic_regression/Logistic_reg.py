import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("score_updated.csv")

def score_to_category(score):
    if score >= 75:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Poor"

df["Performance"] = df["Scores"].apply(score_to_category)

X = df[["Hours"]]  
y = df["Performance"]  

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

joblib.dump(model, "logistic_regression_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("📦 Model saved as 'logistic_regression_classifier.pkl'")
print("📦 Label encoder saved as 'label_encoder.pkl'")
