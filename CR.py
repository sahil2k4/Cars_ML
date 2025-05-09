import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Streamlit settings
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction — Ridge, Lasso, ElasticNet")

# Ignore warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)

# Upload file
uploaded_file = st.file_uploader("📂 Upload your CARS.csv file", type=["csv"])

if uploaded_file:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Handle missing values
    missing_cyl = df["Cylinders"].isnull().sum()
    st.write(f"🔍 Missing values in 'Cylinders': {missing_cyl}")
    df = df.dropna(subset=["Cylinders"])

    # Convert price columns
    df["MSRP"] = df["MSRP"].replace("[$,]", "", regex=True).astype("int64")
    df["Invoice"] = df["Invoice"].replace("[$,]", "", regex=True).astype("int64")

    # Save original for display
    df_display = df.copy()

    # Encode categorical columns
    df_model = df.copy()
    label = LabelEncoder()
    cat_cols = ["Make", "Model", "Type", "Origin", "DriveTrain"]
    for col in cat_cols:
        df_model[col] = label.fit_transform(df_model[col])

    # Features and target
    X = df_model.drop("MSRP", axis=1)
    y = df_model["MSRP"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models and hyperparameters
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
    lasso_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
    elastic_params = {'alpha': [0.01, 0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}

    models = {
        "Ridge": GridSearchCV(Ridge(), ridge_params, cv=5, scoring='r2'),
        "Lasso": GridSearchCV(Lasso(max_iter=10000), lasso_params, cv=5, scoring='r2'),
        "ElasticNet": GridSearchCV(ElasticNet(max_iter=10000), elastic_params, cv=5, scoring='r2')
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        predictions[name] = y_pred
        results[name] = {
            "Best Params": model.best_params_,
            "R² Score": round(r2_score(y_test, y_pred), 4),
            "RMSE": round(mean_squared_error(y_test, y_pred, squared=False), 2)
        }

    # Show model results
    st.subheader("📊 Model Performance")
    st.dataframe(pd.DataFrame(results).T)

    # Plot Actual vs Predicted
    st.subheader("📈 Actual vs Predicted Comparison")
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(predictions.keys()), shared_yaxes=True)

    for i, (name, y_pred) in enumerate(predictions.items(), start=1):
        fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines+markers',
                                 name='Actual', line=dict(color='blue'), showlegend=(i == 1)),
                      row=1, col=i)

        fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines+markers',
                                 name='Predicted', line=dict(color='red'), showlegend=(i == 1)),
                      row=1, col=i)

    fig.update_layout(height=500, width=1200, title_text="Predictions vs Actual")
    st.plotly_chart(fig)

    # Save cleaned file
    df_display.to_csv("carscsv.csv", index=False)
    st.success("✅ Cleaned data saved as `carscsv.csv`")
else:
    st.warning("Please upload your `CARS.csv` file to proceed.")
