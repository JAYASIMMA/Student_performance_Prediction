import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

model = joblib.load("logistic_regression_classifier.pkl")
encoder = joblib.load("label_encoder.pkl")

df = pd.read_csv("score_updated.csv")

def score_to_category(score):
    if score >= 75:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Poor"

df["Performance"] = df["Scores"].apply(score_to_category)

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.markdown("Enter the number of hours studied to predict performance category:")

hours = st.slider("Hours Studied", min_value=0.0, max_value=12.0, step=0.5, value=5.0)
input_df = pd.DataFrame({"Hours": [hours]})

if st.button("Predict"):
    prediction = model.predict(input_df)
    label = encoder.inverse_transform(prediction)[0]
    st.success(f"📊 Predicted Category: **{label}**")

st.header("📈 Data Visualizations")

st.subheader("Category Distribution")
category_counts = df["Performance"].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=["#6dcf9e", "#f6c85f", "#f8665e"])
ax1.axis('equal')
st.pyplot(fig1)

st.subheader("Average Hours by Category")
avg_hours = df.groupby("Performance")["Hours"].mean()
fig2, ax2 = plt.subplots()
sns.barplot(x=avg_hours.index, y=avg_hours.values, ax=ax2, palette="Set2")
ax2.set_ylabel("Average Study Hours")
st.pyplot(fig2)

st.subheader("Hours vs Scores")
fig3, ax3 = plt.subplots()
sns.scatterplot(data=df, x="Hours", y="Scores", hue="Performance", palette="Set1", ax=ax3)
st.pyplot(fig3)

st.subheader("Model Confusion Matrix (on test split)")
from sklearn.model_selection import train_test_split

X = df[["Hours"]]
y = encoder.transform(df["Performance"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
fig4, ax4 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax4)
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")
st.pyplot(fig4)

st.subheader("📋 Classification Report")
report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose().round(2))
