import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Set page configuration
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.title("🚗 Car Price Prediction App with Plotly Visualization")

# Upload dataset
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

    # Distribution histogram
    st.subheader("📊 Distribution of Car Prices")
    fig_hist = px.histogram(df, x="Price", nbins=50, title="Distribution of Car Prices",
                            labels={"Price": "Car Price"}, color_discrete_sequence=['skyblue'])
    fig_hist.update_layout(template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Split data
    X = df_encoded.drop("Price", axis=1)
    y = df_encoded["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse
