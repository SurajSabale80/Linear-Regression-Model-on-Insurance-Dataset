import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("💹 Linear Regression on Insurance Dataset")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your insurance.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Basic stats
    st.subheader("📈 Dataset Information")
    st.write(df.describe())

    # Select target and features
    st.subheader("🎯 Select Features and Target Variable")
    all_columns = df.columns.tolist()
    target_column = st.selectbox("Select target column (Y)", all_columns)
    feature_columns = st.multiselect("Select feature columns (X)", [col for col in all_columns if col != target_column])

    if feature_columns and target_column:
        X = df[feature_columns]
        y = df[target_column]

        # Split data
        test_size = st.slider("Test data size (0.1 to 0.5)", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        st.subheader("📊 Model Performance")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")
        st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.3f}")

        # Coefficients
        st.subheader("📉 Model Coefficients")
        coeff_df = pd.DataFrame(model.coef_, index=feature_columns, columns=["Coefficient"])
        st.write(coeff_df)

        # Visualization
        st.subheader("📈 Actual vs Predicted Plot")
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted Values")
        st.pyplot(fig)

        # Correlation heatmap
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

else:
    st.info("👆 Please upload your insurance.csv file to get started.")
