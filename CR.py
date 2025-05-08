import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Example: After prediction is done
if st.button("Predict Price"):
    # Assume y_test and y_pred are available from your model
    st.subheader("🔍 Actual vs Predicted Car Prices")
    
    # Combine actual and predicted for plotting
    result_df = pd.DataFrame({
        "Actual Price": y_test,
        "Predicted Price": y_pred
    }).reset_index(drop=True)

    # Plot using Plotly for interactivity
    fig = px.scatter(result_df, x="Actual Price", y="Predicted Price",
                     title="Actual vs Predicted Car Prices",
                     labels={"Actual Price": "Actual", "Predicted Price": "Predicted"},
                     color_discrete_sequence=["#00cc96"])
    
    fig.add_shape(
        type="line", x0=result_df["Actual Price"].min(), y0=result_df["Actual Price"].min(),
        x1=result_df["Actual Price"].max(), y1=result_df["Actual Price"].max(),
        line=dict(color="red", dash="dash")
    )
    st.plotly_chart(fig, use_container_width=True)
