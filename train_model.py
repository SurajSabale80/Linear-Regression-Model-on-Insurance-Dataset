import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("insurance.csv")

# Encode categorical variables
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["region"] = df["region"].map({"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3})

# Features & Target
X = df[["age", "bmi", "children", "smoker", "region"]]
y = df["charges"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save trained model
with open("base_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as base_model.pkl")
