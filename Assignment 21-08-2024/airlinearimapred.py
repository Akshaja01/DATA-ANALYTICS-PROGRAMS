
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from pmdarima import auto_arima

st.set_page_config(page_title= "Time Series Analysis", page_icon=":small_airplane:", layout="wide")
st.title(":airplane_departure: Airline -ARIMA Data Forecasting:small_airplane:")


df = pd.read_csv("airline_passengers.csv", index_col = "Month", parse_dates = True)
st.header("Airline Dataset")
st.dataframe(df.head())


train = df[0:120]
test  = df[120:]


st.header("Min Value")
result = pd.read_csv("arima_order.csv")
min1 = result[result.RMSE==result.RMSE.min()]
min_order_val = min1['order']
st.write(min_order_val)
min_order_val = (9,0,8) #===
min_order_string = ''.join(str(min_order_val))


model1 = ARIMA(train, order=(4,1,3))
model1.initialize_approximate_diffuse()
model1_arima = model1.fit()
test_arima1=model1_arima.predict(start=121, end=144)



model2 = ARIMA(train,order=min_order_val)
model2.initialize_approximate_diffuse()
model2_arima=model2.fit()
test_arima2=model2_arima.predict(start=121, end=144)



plt.figure(figsize=(10,8))
plt.title("1. Forecasting using ARIMA(4,1,3)")
plt.plot(train['Passengers'],color='r',label='original data')
plt.plot(test['Passengers'],color='g',label='tested data')
plt.plot(test_arima1,color='b',label='ARIMA data (4,1,3)')
plt.xlabel('Years')
plt.ylabel('Passengers')
plt.legend()
st.pyplot(plt.gcf())



plt.figure(figsize=(10,8))
plt.title("2. Forecasting using ARIMA "+ min_order_string)
plt.plot(train['Passengers'],color='r',label='original data')
plt.plot(test['Passengers'],color='g',label='tested data')
plt.plot(test_arima2,color='b',label='ARIMA data '+ min_order_string)
plt.xlabel('Years')
plt.ylabel('Passengers')
plt.legend()
st.pyplot(plt.gcf())




