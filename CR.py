import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Set page config
st.set_page_config(page_title="Car Price Predictor", layout="wide", initial_sidebar_state="expanded")

st.title("🚗 Car Price Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your car dataset CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "Price" not in df.columns:
        st.error("Column 'Price' not found. Please upload a dataset with a 'Price' column.")
        st.stop()

    # Encode categorical columns
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    # Plotly histogram
    st.subheader("📊 Distribution of Car Prices")
    fig = px.histogram(df, x="Price", nbins=50, title="Distribution of Car Prices",
                       labels={"Price": "Car Price"}, color_discrete_sequence=['skyblue'])
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Split data
    X = df_encoded.drop("Price", axis=1)
    y = df_encoded["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.subheader("📈 Model Performance")
    st.write(f"Root Mean Squared Error: **{rmse:.2f}**")

    # Make prediction using user input
    st.subheader("🔍 Predict Car Price")
    input_data = {}
    for col in X.columns:
        if df[col].dtype == "object":
            input_data[col] = st.selectbox(f"Select {col}", sorted(df[col].unique()))
        else:
            input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))

    # Prepare input
    input_df = pd.DataFrame([input_data])
    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = le.fit_transform(input_df[col].astype(str))

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Car Price: **{prediction:,.2f}**")

else:
    st.info("Please upload a CSV file to begin.")
