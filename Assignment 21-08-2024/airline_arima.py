import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import itertools
import numpy as np

st.set_page_config(page_title="Airline data analysis", page_icon=":small_airplane:", layout="wide")
st.title(":small_airplane: Airline Data Analysis :small_airplane:")

df = pd.read_csv("airline_passengers.csv", index_col = "Month", parse_dates = True)

st.header("Airline Dataset")
st.dataframe(df.head())

st.line_chart(data = df, y = "Passengers")


components = seasonal_decompose(df['Passengers'], model = "multiplicative")
fig1 = components.plot()
st.pyplot(fig1)

st.divider()


def train_test(df):
    train = df[0:120]
    test  = df[120:]
    return train,test


tr1, ts1   = train_test(df)
model_ar1 = auto_arima(tr1,start_p=1,start_q=1,max_p=10,max_q=10,m=1,d=1,seasonal=True,trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)


st.header('Best parameters are:')
st.write(model_ar1.summary())


p=range(0,10)
d=range(0,2)
q=range(0,10)

pdq_combination = list(itertools.product(p,d,q))

rmse   = []
order1 = []

for pdq in pdq_combination:
 
    model = ARIMA(tr1, order = pdq )
    model.initialize_approximate_diffuse()  
    model_ar2 = model.fit()

    pred      = model_ar2.predict(start=121, end=144)
    error     = np.sqrt(mean_squared_error(ts1,pred))
    order1.append(pdq)
    rmse.append(error)
    
result = pd.DataFrame({'order':order1, 'RMSE': rmse})
result.to_csv("arima_order.csv")


