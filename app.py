import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("💹 Insurance Cost")

st.write("""
This app predicts **Medical Insurance Costs** using a pre-trained Linear Regression model.
Enter your details below to estimate your insurance charges.
""")

# Load trained model
try:
    with open("base_model.pkl", "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ base_model.pkl not found! Please train or add your model file.")
    st.stop()

# Sidebar inputs
st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 1)
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Convert categorical features
smoker_val = 1 if smoker == "yes" else 0
region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
region_val = region_map[region]

# Prepare input for prediction
input_data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker_val],
    "region": [region_val]
})

# Prediction
if st.button("Predict Insurance Cost"):
    prediction = model.predict(input_data)
    st.success(f"💰 **Predicted Insurance Cost:** ${prediction[0]:,.2f}")

# Optional visualization (for demonstration)
st.subheader("📊 Example Visualization (Model Coefficients)")
if hasattr(model, "coef_"):
    coef_df = pd.DataFrame({
        "Feature": ["age", "bmi", "children", "smoker", "region"],
        "Coefficient": model.coef_
    })
    fig, ax = plt.subplots()
    sns.barplot(x="Feature", y="Coefficient", data=coef_df, ax=ax)
    plt.title("Linear Regression Coefficients")
    st.pyplot(fig)
else:
    st.info("Model coefficients not available.")

