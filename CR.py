# Streamlit app for car price prediction

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("CARS.csv")  # Make sure this CSV is in the same folder
    return df

df = load_data()

# Title
st.title("🚗 Car Price Prediction App")

# Show Data
if st.checkbox("Show Raw Dataset"):
    st.write(df)

# Preprocessing
df = df.dropna()
label_encoders = {}
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# User Input
st.sidebar.header("📥 Input Car Features")
user_input = {}

for col in X.columns:
    if col in categorical_cols:
        options = label_encoders[col].classes_
        selected = st.sidebar.selectbox(f"{col}", options)
        encoded = label_encoders[col].transform([selected])[0]
        user_input[col] = encoded
    else:
        value = st.sidebar.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        user_input[col] = value

input_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Car Price: ${prediction:,.2f}")

# Visualization
st.subheader("🔍 Price Distribution by Car Company")
if "Company" in df.columns:
    fig = px.box(df, x="Company", y="Price", title="Car Price Distribution by Company")
    st.plotly_chart(fig)
