import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split as tts
from sklearn import metrics as mat

st.set_page_config(page_title="Diamond Data Analysis", page_icon="💎", layout="wide")
st.title("💎 Diamond Data Analysis 💎")


df = pd.read_csv('diamonds.csv')
st.header('Diamonds Dataset')
st.table(df.head())


st.header("Statistical Summary")
st.table(df.describe())


st.header("Data Visualization")


fig1 = px.scatter(df, x="carat", y="price", color="cut", color_continuous_scale='Viridis')
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Price Distribution")
fig2 = px.histogram(df, x="price", nbins=20)
st.plotly_chart(fig2, use_container_width=True)


st.subheader("Boxplot for Price with Cuts")
fig3 = px.box(df, x="cut", y="price", color="cut")
st.plotly_chart(fig3, use_container_width=True)

cat_cols = ['cut', 'clarity', 'color']
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

st.header("Updated Diamonds Dataset")
st.table(df.head())


x = df.drop(columns=['price'], axis=1)
y = df[['price']]

xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2, random_state=10, shuffle=True)

gbr = GradientBoostingRegressor(max_depth=2, learning_rate=0.1)
gbr.fit(xtrain, ytrain)
ypred1 = gbr.predict(xtest)


pickle.dump(gbr, open('gbr.pkl', 'wb'))


adr = AdaBoostRegressor()
adr.fit(xtrain, ytrain)
ypred2 = adr.predict(xtest)


pickle.dump(adr, open('adr.pkl', 'wb'))


xgb = XGBRegressor()
xgb.fit(xtrain, ytrain)
ypred3 = xgb.predict(xtest)


pickle.dump(xgb, open('xgb.pkl', 'wb'))

cat = CatBoostRegressor(iterations=100, learning_rate=0.1, loss_function='RMSE', verbose=0)
cat.fit(xtrain, ytrain)
ypred4 = cat.predict(xtest)


pickle.dump(cat, open('cat.pkl', 'wb'))


st.header("Comparison of R2 Scores of Different Models")
c1, c2, c3, c4 = st.columns(4)

c1.subheader("R2 Score of Gradient Boosting")
c1.subheader(round(mat.r2_score(ytest, ypred1), 2))

c2.subheader("R2 Score of AdaBoost")
c2.subheader(round(mat.r2_score(ytest, ypred2), 2))

c3.subheader("R2 Score of XGBoost")
c3.subheader(round(mat.r2_score(ytest, ypred3), 2))

c4.subheader("R2 Score of CatBoost")
c4.subheader(round(mat.r2_score(ytest, ypred4), 2))