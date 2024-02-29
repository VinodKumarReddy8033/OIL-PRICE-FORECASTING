import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
import datetime as dt
import yfinance as yf
from plotly.subplots import make_subplots
from datetime import timedelta
from datetime import date
import pickle
loaded_model=pickle.load(open("prophet.sav",'rb'))


st.title('Oil Price Preduction')

start_date=st.sidebar.date_input("Start Date", pd.to_datetime("today") - pd.DateOffset(days=365))
forecast_days = st.sidebar.slider("Number of Days to Forecast", 1, 365, 30)
# Generate date range for forecasting
future_dates = pd.date_range(start=start_date, periods=forecast_days)
# Predict using the Prophet model
future = pd.DataFrame(future_dates, columns=['ds'])
forecast = loaded_model.predict(future)
# Display the forecasted data
st.subheader('Forecasted Oil Prices:')
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
# Visualize the forecast
fig = loaded_model.plot(forecast)
st.write(fig)
 #Show components (trends and seasonality)
fig_comp = loaded_model.plot_components(forecast)
st.write(fig_comp)
