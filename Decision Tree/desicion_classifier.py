import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# === Step 1: Load Dataset ===
df = pd.read_csv("score_updated.csv")

# === Step 2: Convert Scores to Categories ===
def categorize(score):
    if score >= 75:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Poor"

df["Performance"] = df["Scores"].apply(categorize)

# === Step 3: Feature and Target Separation ===
X = df.drop(columns=["Scores", "Performance"])
y = df["Performance"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === Step 4: Train Decision Tree Classifier ===
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# === Step 5: Predictions and Evaluation ===
y_pred = clf.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === Step 6: Save the Model and Encoder ===
joblib.dump(clf, "decision_tree_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("📦 Model and encoder saved as 'decision_tree_classifier.pkl' and 'label_encoder.pkl'")
