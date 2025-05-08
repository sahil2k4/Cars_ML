import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🚗 Car Price Prediction - Model Performance Visualizer")

# Load and preprocess data
df = pd.read_csv("DataSets/CARS.csv")
df.dropna(subset=["Cylinders"], inplace=True)

df["MSRP"] = df["MSRP"].replace("[$,]", "", regex=True).astype(int)
df["Invoice"] = df["Invoice"].replace("[$,]", "", regex=True).astype(int)

label = LabelEncoder()
for col in ["Make", "Model", "Type", "Origin", "DriveTrain"]:
    df[col] = label.fit_transform(df[col])

X = df.drop("MSRP", axis=1)
y = df["MSRP"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and predict
results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    results[name] = {
        "y_pred": y_pred,
        "r2": r2_score(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred, squared=False)
    }

# Create subplots for comparison
actual = pd.Series(y_test).reset_index(drop=True)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=list(results.keys()),
    shared_xaxes=True, shared_yaxes=True
)

model_colors = {
    "Linear Regression": "red",
    "Random Forest": "green",
    "Decision Tree": "orange",
    "Gradient Boosting": "purple"
}

positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

for (name, res), (row, col) in zip(results.items(), positions):
    fig.add_trace(go.Scatter(
        y=actual,
        mode='lines+markers',
        name=f"{name} - Actual" if (row, col) == (1, 1) else "",
        line=dict(color="blue")
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        y=res["y_pred"],
        mode='lines+markers',
        name=f"{name} - Predicted" if (row, col) == (1, 1) else "",
        line=dict(color=model_colors[name])
    ), row=row, col=col)

fig.update_layout(
    title="📊 Actual vs Predicted — All Models",
    height=900,
    width=1100,
    showlegend=True
)

fig.update_xaxes(title_text="Test Sample Index")
fig.update_yaxes(title_text="Target Value")

st.plotly_chart(fig, use_container_width=True)

# Display R² and RMSE Table
metrics_data = {
    "Model": [],
    "R² Score": [],
    "RMSE": []
}
for name, res in results.items():
    metrics_data["Model"].append(name)
    metrics_data["R² Score"].append(round(res["r2"], 4))
    metrics_data["RMSE"].append(round(res["rmse"], 2))

metrics_df = pd.DataFrame(metrics_data)
st.subheader("📈 Model Evaluation Metrics")
st.dataframe(metrics_df, use_container_width=True)
