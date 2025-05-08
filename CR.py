import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Car Price Prediction App")

uploaded_file = st.file_uploader("📂 Upload your CARS dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded and loaded!")

    st.subheader("🧾 Available Columns")
    st.write(df.columns.tolist())

    # Ask user to select the target (price) column
    target_column = st.selectbox("🎯 Select the Price Column (Target)", df.columns)

    # Show raw data
    if st.checkbox("🔍 Show Raw Data"):
        st.dataframe(df)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Encode categorical columns
    label_encoders = {}
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Split features and target
    try:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
    except KeyError:
        st.error(f"❌ The selected column '{target_column}' does not exist. Please re-upload your data.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

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

    input_df = pd.DataFrame([input_data])
    if st.button("🔮 Predict Car Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Car Price: ${prediction:,.2f}")

    st.subheader("📊 Price Distribution")
    fig = px.histogram(df, x=target_column, nbins=30, title="Distribution of Car Prices")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CARS.csv file to get started.")
