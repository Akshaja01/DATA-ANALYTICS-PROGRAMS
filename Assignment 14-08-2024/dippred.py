import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics as mat
import pickle

st.set_page_config(page_title="Diamond Data Analysis", page_icon="💎", layout="wide")
st.title("💎 Diamond Data Analysis 💎")


df = pd.read_csv('diamonds.csv')
st.header('Diamond Dataset')
st.table(df.head())


st.header("Statistical Summary")
st.table(df.describe())


st.header("Data Visualization")
fig1 = px.scatter(df, x="carat", y="price", color="cut", color_continuous_scale='Viridis')
st.plotly_chart(fig1, use_container_width=True)

st.header("Price Distribution")
fig2 = px.histogram(df, x="price", nbins=20)
st.plotly_chart(fig2, use_container_width=True)

st.header("Box Plot for Price with Cuts")
fig3 = px.box(df, x="cut", y="price", color="cut")
st.plotly_chart(fig3, use_container_width=True)


cat_col = ['cut', 'clarity', 'color']
le = LabelEncoder()
for col in cat_col:
    df[col] = le.fit_transform(df[col])
st.header("Updated Diamond Dataset")
st.table(df.head())


x = df.drop(columns=['price'])
y = df['price']


c1, c2 = st.columns(2)
c1.subheader("Features")
c1.table(x.head())
c2.subheader("Labels")
c2.table(y.head())


xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2, random_state=10, shuffle=True)


c3, c4, c5, c6 = st.columns(4)
c3.subheader("Training Features Size")
c3.table(xtrain.head())
c4.subheader("Training Labels Size")
c4.table(ytrain.head())
c5.subheader("Testing Features Size")
c5.table(xtest.head())
c6.subheader("Testing Labels Size")
c6.table(ytest.head())


gbr = GradientBoostingRegressor(max_depth=2, n_estimators=5, learning_rate=1.0)
gbr.fit(xtrain, ytrain)
ypred1 = gbr.predict(xtest)
r1 = mat.r2_score(ytest, ypred1)
pickle.dump(gbr, open('gbr1.pkl', 'wb'))

adr = AdaBoostRegressor()
adr.fit(xtrain, ytrain)
ypred2 = adr.predict(xtest)
r2 = mat.r2_score(ytest, ypred2)
pickle.dump(adr, open('adr1.pkl', 'wb'))

#xgb = XGBRegressor()
#xgb.fit(xtrain, ytrain)
#ypred3 = xgb.predict(xtest)
#r3 = mat.r2_score(ytest, ypred3)
#pickle.dump(xgb, open('xgb1.pkl', 'wb'))

st.header("Model Comparison")
cl1, cl2, cl3 = st.columns(3)
cl1.subheader('Gradient Boosting R2 Score')
cl1.write(r1)
cl2.subheader('AdaBoost R2 Score')
cl2.write(r2)
#cl3.subheader('XGBoost R2 Score')
#cl3.write(r3)