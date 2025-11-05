import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="Insurance Cost", page_icon="💹")
st.title("💹 Insurance Cost Prediction App")

st.write("""
This Streamlit app predicts **Medical Insurance Costs** using a pre-trained **Linear Regression** model.
Enter your personal details in the sidebar to estimate your insurance charges.
""")

# --- LOAD MODEL ---
try:
    with open("best_model1.pkl", "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model 'best_model1.pkl' loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file 'best_model1.pkl' not found. Please upload or train the model first.")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Input Features")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 1)
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# --- ENCODE INPUTS ---
smoker_val = 1 if smoker == "yes" else 0
region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
region_val = region_map[region]

# --- PREPARE INPUT DATA ---
input_data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker_val],
    "region": [region_val]
})

# Ensure input columns match model training columns
try:
    input_data = input_data[model.feature_names_in_]
except Exception:
    st.warning("⚠️ Model does not store feature names. Using default column order.")

# --- PREDICTION ---
if st.button("🔮 Predict Insurance Cost"):
    try:
        prediction = model.predict(input_data)
        st.success(f"💰 **Predicted Insurance Cost:** ${prediction[0]:,.2f}")
    except ValueError as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

# --- MODEL COEFFICIENT VISUALIZATION ---
st.subheader("📊 Linear Regression Coefficients")

try:
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = ["age", "bmi", "children", "smoker", "region"]

    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
    })

    st.dataframe(coef_df)

    fig, ax = plt.subplots()
    sns.barplot(x="Feature", y="Coefficient", data=coef_df, ax=ax)
    plt.title("Feature Importance in Linear Regression")
    plt.xticks(rotation=30)
    st.pyplot(fig)

except Exception as e:
    st.warning(f"⚠️ Could not display coefficients: {e}")

st.markdown("---")
st.markdown("👨‍💻 **Developed by Suraj Sabale** | 📈 *Insurance Cost Prediction App*")

