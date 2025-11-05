import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# APP TITLE AND DESCRIPTION
# ----------------------------
st.set_page_config(page_title="Insurance Cost", page_icon="💹")
st.title("💹 Insurance Cost")
st.write("""
This app predicts **Medical Insurance Costs** using a pre-trained **Linear Regression** model.
Enter your details in the sidebar to estimate your insurance charges.
""")

# ----------------------------
# LOAD TRAINED MODEL
# ----------------------------
try:
    with open("base_model.pkl", "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ 'base_model.pkl' not found! Please train your model using train_model.py.")
    st.stop()

# ----------------------------
# SIDEBAR - USER INPUTS
# ----------------------------
st.sidebar.header("Input Features")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 1)
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Convert categorical values to numeric
smoker_val = 1 if smoker == "yes" else 0
region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
region_val = region_map[region]

# Prepare data for prediction
input_data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker_val],
    "region": [region_val]
})

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("🔮 Predict Insurance Cost"):
    prediction = model.predict(input_data)
    st.success(f"💰 **Predicted Insurance Cost:** ${prediction[0]:,.2f}")

# ----------------------------
# MODEL COEFFICIENTS VISUALIZATION
# ----------------------------
st.subheader("📊 Model Coefficients")

try:
    # Automatically detect feature names if available
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        # Fallback to generic names
        features = [f"Feature {i+1}" for i in range(len(model.coef_))]

    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
    })

    st.write(coef_df)

    fig, ax = plt.subplots()
    sns.barplot(x="Feature", y="Coefficient", data=coef_df, ax=ax)
    plt.title("Linear Regression Coefficients")
    plt.xticks(rotation=30)
    st.pyplot(fig)

except Exception as e:
    st.warning(f"⚠️ Could not display coefficients: {e}")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("👨‍💻 **Developed by Suraj Sabale** | 📊 *Insurance Cost Prediction App*")
