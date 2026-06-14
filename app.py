import os
import joblib
import streamlit as st
import pandas as pd


st.set_page_config(page_title="sales prediction",page_icon=":chart_with_upwards_trend:",layout="wide")

st.write("Enter the details of TV,Radio,and Newspaper advertising budgets to predict sales")

model=joblib.load("models/linear_regression_model.pkl")

tv_budget=st.number_input("TV Advertising Budget",min_value=0.0,step=0.01)
radio_budget=st.number_input("Radio Advertising Budget",min_value=0.0,step=0.01)
newspaper_budget=st.number_input("Newspaper Advertising Budget",min_value=0.0,step=0.01)
if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        "TV":[tv_budget],
        "Radio":[radio_budget],
        "Newspaper":[newspaper_budget]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Sales: {prediction:.2f} units")
