import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import SimpleExpSmoothing,ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf

st.set_page_config(page_title= "Aireline data analysis", page_icon=":small_airplane:", layout="wide")
st.title(":airplane_departure: Airline Data Analysis :small_airplane:")


df = pd.read_csv("airline_passengers.csv", index_col = "Month", parse_dates = True)

st.header("Airline Dataset")
st.dataframe(df.head())

st.line_chart(data = df, y = "Passengers")


components = seasonal_decompose(df['Passengers'], model = "multiplicative")
fig1 = components.plot()
st.pyplot(fig1)

#============================================================
st.header("Simple average method")

df1=df.copy()
ma_window=12
df1['ma']=df1['Passengers'].rolling(ma_window).mean()
st.header("The new data with averaging technique")
st.dataframe(df1)

st.subheader("Replace the first 12 values with previous values")
df1['ma'][0:12]=df1['Passengers'][0:12]
st.dataframe(df1)

plt.figure(figsize=(20,5))
plt.plot(df1['Passengers'],color='r',label='originaldata')
plt.plot(df1['ma'],color='g',label='Predicted data')
plt.xlabel('Years')
plt.ylabel('Passengers')

plt.legend()
st.pyplot(plt.gcf())
alpha=1/(2*(ma_window))
df1['Simple_exp']=SimpleExpSmoothing(df1['Passengers']).fit(smoothing_level=alpha).fittedvalues

plt.figure(figsize=(20,5))
plt.plot(df1['Passengers'],color='r',label='originaldata')
plt.plot(df1['ma'],color='g',label='Predicted data')
plt.plot(df1['Simple_exp'],color='b',label='Predicted data using simple exponential method')

plt.xlabel('Years')
plt.ylabel('Passengers')

plt.legend()
st.pyplot(plt.gcf())
#===========================================================

st.subheader("Forecasting using simple exp smoothing")
train=df[0:120]
test=df[120:]
c1,c2=st.columns(2)

c1.header("Training data")
c1.subheader(train.shape)

c2.header("Testing data")
c2.subheader(test.shape)

simpleexpmodel=SimpleExpSmoothing(train['Passengers']).fit(smoothing_level=alpha,optimized=True,use_brute=True)

test_pred1=simpleexpmodel.forecast(24)

plt.figure(figsize=(20,5))
plt.plot(train['Passengers'],color='r',label='original data')
plt.plot(test['Passengers'],color='g',label='Predicted data')
plt.plot(test_pred1,color='b',label='Predicted data using simple exponential method')

plt.xlabel('Years')
plt.ylabel('Passengers')

plt.legend()
st.pyplot(plt.gcf())

#====================================================================
exptmodel1=ExponentialSmoothing(train['Passengers'],trend='add').fit()

exptmodel2=ExponentialSmoothing(train['Passengers'],trend='mul').fit()

test_pred2=exptmodel1.forecast(24)
test_pred3=exptmodel2.forecast(24)

st.header("Forecasting using exp smoothing trend addition")

plt.figure(figsize=(20,5))
plt.plot(train['Passengers'],color='r',label='original data')
plt.plot(test['Passengers'],color='g',label='Predicted data')
plt.plot(test_pred2,color='b',label='Predicted data using simple exponential smoothing trend addition method')

plt.xlabel('Years')
plt.ylabel('Passengers')

plt.legend()
st.pyplot(plt.gcf())
#====================================================

st.header("Forecasting using exp smoothing trend multiplication")

plt.figure(figsize=(20,5))
plt.plot(train['Passengers'],color='r',label='original data')
plt.plot(test['Passengers'],color='g',label='Predicted data')
plt.plot(test_pred3,color='b',label='Predicted data using simple exponential smoothing trend multiplication method')

plt.xlabel('Years')
plt.ylabel('Passengers')

plt.legend()
st.pyplot(plt.gcf())

# ====================================================================

exptmodel3=ExponentialSmoothing(train['Passengers'],trend='add',seasonal='add').fit()
exptmodel4=ExponentialSmoothing(train['Passengers'],trend='mul',seasonal='mul').fit()


test_pred4=exptmodel3.forecast(24)
test_pred5=exptmodel4.forecast(24)


plt.figure(figsize=(20,5))
plt.title("Forecasting using Exponential Smoothing with Trend & Season Addition")
plt.plot(train['Passengers'], color='r', label="Original Data")
plt.plot(test['Passengers'], color='g', label="Test Data")
plt.plot(test_pred4, color='b', label="Forecasting using Exponential Smoothing with Trend & Season Addition")
plt.xlabel('Years')
plt.ylabel('Passengers')
plt.legend()
st.pyplot(plt.gcf())

plt.figure(figsize=(20,5))
plt.title("Forecasting using Exponential Smoothing with Trend & Season Multiplication")
plt.plot(train['Passengers'], color='r', label="Original Data")
plt.plot(test['Passengers'], color='g', label="Test Data")
plt.plot(test_pred5, color='b', label="Forecasting using Exponential Smoothing with Trend & Season Multiplication")
plt.xlabel('Years')
plt.ylabel('Passengers')
plt.legend()
st.pyplot(plt.gcf())

#===================================

model=ARIMA(train,order=(1,0,0))
model1=model.fit()

test_arima1=model1.forecast(24)
plt.figure(figsize=(20,5))
plt.title("Forecasting using arima(1,0,0)")
plt.plot(train['Passengers'], color='r', label="Original Data")
plt.plot(test['Passengers'], color='g', label="Test Data")
plt.plot(test_arima1, color='b', label="Forecasting using arima(1,0,0)")
plt.xlabel('Years')
plt.ylabel('Passengers')
plt.legend()
st.pyplot(plt.gcf())
#===========================

model_q=ARIMA(train,order=(0,0,12))
model2=model_q.fit()

test_arima2=model2.forecast(24)
plt.figure(figsize=(20,5))
plt.title("Forecasting using arima(0,0,12)")
plt.plot(train['Passengers'], color='r', label="Original Data")
plt.plot(test['Passengers'], color='g', label="Test Data")
plt.plot(test_arima2, color='b', label="Forecasting using arima(0,0,12)")
plt.xlabel('Years')
plt.ylabel('Passengers')
plt.legend()
st.pyplot(plt.gcf())

#=========================

model_d=ARIMA(train,order=(0,1,0))
model3=model_d.fit()

test_arima3=model3.forecast(24)
plt.figure(figsize=(20,5))
plt.title("Forecasting using arima(0,1,0)")
plt.plot(train['Passengers'], color='r', label="Original Data")
plt.plot(test['Passengers'], color='g', label="Test Data")
plt.plot(test_arima3, color='b', label="Forecasting using arima(0,1,0)")
plt.xlabel('Years')
plt.ylabel('Passengers')
plt.legend()
st.pyplot(plt.gcf())


#===============================================

st.header("Auto correlation plot with airline data")
st.pyplot(plot_acf(df).figure)