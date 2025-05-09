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

# Streamlit page config
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction using Ridge, Lasso, ElasticNet")

# Ignore warnings
warnings.filterwarnings("ignore")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your CARS.csv file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")
        st.subheader("🔍 Preview of Dataset")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # Drop missing 'Cylinders'
    df = df.dropna(subset=["Cylinders"])

    # Clean currency columns
    try:
        df["MSRP"] = df["MSRP"].replace("[$,]", "", regex=True).astype("int64")
        df["Invoice"] = df["Invoice"].replace("[$,]", "", regex=True).astype("int64")
    except Exception as e:
        st.error(f"Currency column error: {e}")
        st.stop()

    # Encode categorical columns
    df_encoded = df.copy()
    label_enc = LabelEncoder()
    for col in ["Make", "Model", "Type", "Origin", "DriveTrain"]:
        df_encoded[col] = label_enc.fit_transform(df_encoded[col])

    # Split X and y
    X = df_encoded.drop("MSRP", axis=1)
    y = df_encoded["MSRP"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model definitions
    models = {
        "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
        "Lasso": (Lasso(max_iter=10000), {"alpha": [0.01, 0.1, 1, 10, 100]}),
        "ElasticNet": (ElasticNet(max_iter=10000), {
            "alpha": [0.01, 0.1, 1, 10],
            "l1_ratio": [0.1, 0.5, 0.9]
        })
    }

    # Train and evaluate
    results = {}
    predictions = {}
    for name, (model, params) in models.items():
        grid = GridSearchCV(model, params, cv=5, scoring='r2')
        grid.fit(x_train, y_train)
        y_pred = grid.predict(x_test)
        predictions[name] = y_pred
        results[name] = {
            "R²": round(r2_score(y_test, y_pred), 4),
            "RMSE": round(mean_squared_error(y_test, y_pred, squared=False), 2),
            "Best Params": grid.best_params_
        }

    # Display results
    st.subheader("📊 Model Performance")
    st.dataframe(pd.DataFrame(results).T)

    # Plot predictions
    st.subheader("📈 Actual vs Predicted")
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(predictions.keys()), shared_yaxes=True)

    for i, (name, pred) in enumerate(predictions.items(), start=1):
        fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test,
                                 mode='lines+markers', name='Actual',
                                 line=dict(color='blue'), showlegend=(i == 1)), row=1, col=i)
        fig.add_trace(go.Scatter(x=list(range(len(pred))), y=pred,
                                 mode='lines+markers', name='Predicted',
                                 line=dict(color='red'), showlegend=(i == 1)), row=1, col=i)

    fig.update_layout(height=500, width=1100, title_text="📉 Actual vs Predicted: Ridge | Lasso | ElasticNet")
    st.plotly_chart(fig)

    # Export cleaned CSV
    df_encoded.to_csv("carscsv.csv", index=False)
    st.success("📦 Cleaned file saved as `carscsv.csv`")

else:
    st.info("📤 Please upload your `CARS.csv` file to begin.")
