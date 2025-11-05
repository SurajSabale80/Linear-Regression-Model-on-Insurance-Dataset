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

# --- SHOW MODEL FEATURE NAMES (DEBUG SECTION) ---
if hasattr(model, "feature_names_in_"):
    st.info(f"🧠 Model was trained with these features:\n\n{list(model.feature_names_in_)}")
else:
    st.warning("⚠️ Model does not store feature names. Proceeding with manual input order.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Input Features")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 1)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# --- ENCODE INPUTS ---
smoker_val = 1 if smoker == "yes" else 0

# Create dummy variables like training
input_dict = {
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker_val],
    "sex_male": [1 if sex == "male" else 0],
    "region_northwest": [1 if region == "northwest" else 0],
    "region_southeast": [1 if region == "southeast" else 0],
    "region_southwest": [1 if region == "southwest" else 0],
}

input_data = pd.DataFrame(input_dict)

# Align columns with model training
try:
    if hasattr(model, "feature_names_in_"):
        missing = set(model.feature_names_in_) - set(input_data.columns)
        for col in missing:
            input_data[col] = 0  # Add missing columns as 0 (default)
        input_data = input_data[model.feature_names_in_]
except Exception as e:
    st.warning(f"⚠️ Could not align features automatically: {e}")

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
        features = [f"Feature {i+1}" for i in range(len(model.coef_))]

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

