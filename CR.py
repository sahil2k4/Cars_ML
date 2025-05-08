import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# App Title
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction App")

# File Uploader
uploaded_file = st.file_uploader("📂 Upload your CARS.csv file", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Show raw data
    if st.checkbox("Show Raw Dataset"):
        st.dataframe(df)

    # Drop missing values
    df.dropna(inplace=True)

    # Encode categorical columns
    label_encoders = {}
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Split data
    X = df.drop("Price", axis=1)
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Sidebar for user input
    st.sidebar.header("📥 Input Car Features")
    input_data = {}

    for col in X.columns:
        if col in categorical_cols:
            options = label_encoders[col].classes_
            selected = st.sidebar.selectbox(f"{col}", options)
            input_data[col] = label_encoders[col].transform([selected])[0]
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            input_data[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)

    # Predict Price
    input_df = pd.DataFrame([input_data])
    if st.button("🔮 Predict Car Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")

    # Visualization
    st.subheader("📊 Car Price Distribution by Company")
    if "Company" in df.columns:
        fig = px.box(df, x="Company", y="Price", title="Car Price by Company")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CARS.csv file to start.")
