import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("xgb_score_model.pkl")

df = pd.read_csv("score_updated.csv")

# Title
st.title("🎓 Student Performance Predictor")

# Input: Study Hours
st.sidebar.header("Input Student Study Hours")
hours = st.sidebar.slider("Study Hours", min_value=0.0, max_value=10.0, step=0.1)

# Prediction
input_df = pd.DataFrame({
    "Hours": [hours],
    "Category": ["Default"]  # required dummy categorical column
})

predicted_score = model.predict(input_df)[0]

# Classify performance
if predicted_score >= 75:
    performance = "Good"
    color = "green"
elif predicted_score >= 50:
    performance = "Average"
    color = "orange"
else:
    performance = "Bad"
    color = "red"

# Display result
st.markdown(f"### Predicted Score: **{predicted_score:.2f}**")
st.markdown(f"### Performance: :{color}[**{performance}**]")

# Visualizations
st.header("📊 Data Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"])

with tab1:
    st.subheader("Study Hours vs Scores")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x="Hours", y="Scores", hue=(df['Scores'] >= 50), palette={True: "green", False: "red"}, s=100)
    ax1.axhline(50, color='gray', linestyle='--', label='Passing Score')
    ax1.set_title("Scatter Plot of Study Hours vs Scores")
    st.pyplot(fig1)

with tab2:
    st.subheader("Score Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['Scores'], kde=True, bins=10, color="skyblue")
    ax2.set_title("Histogram of Scores")
    st.pyplot(fig2)

with tab3:
    st.subheader("Box Plot of Scores")
    fig3, ax3 = plt.subplots()
    sns.boxplot(y=df['Scores'], color="lightgreen")
    ax3.set_title("Box Plot of Scores")
    st.pyplot(fig3)

with tab4:
    st.subheader("Correlation Heatmap")
    fig4, ax4 = plt.subplots()
    corr = df[['Hours', 'Scores']].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
    ax4.set_title("Correlation between Hours and Scores")
    st.pyplot(fig4)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit & XGBoost")
