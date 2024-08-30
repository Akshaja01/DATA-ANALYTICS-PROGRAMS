import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics as mat
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



st.set_page_config(page_title="Diabetes Data Analysis", page_icon="🍦", layout="wide")
st.title("🍩 Diabetes Data Analysis 🍩️")


df=pd.read_csv('diabetes.csv')
st.header('🍬DIABETES DATA SET🍬')
st.table(df.head())

st.header('🍰STATISTICAL SUMMARY OF DIABETES DATA SET🍰')
st.table(df.describe())

x=df.drop(columns=['Outcome'],axis=1)
y=df[['Outcome']]

c1,c2=st.columns(2)

c1.header("Features set")
c1.table(x.head())

c2.header("Labels")
c2.table(y.head())

xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2, random_state=10, shuffle=True, stratify=y)

c3,c4,c5,c6=st.columns(4)

c3.header("Training Features set")
c3.subheader(xtrain.shape)
c3.table(xtrain.head())

c4.header("Training Labels")
c4.subheader(ytrain.shape)
c4.table(ytrain.head())

c5.header("Testing Features set")
c5.subheader(xtest.shape)
c5.table(xtest.head())

c6.header("Testing Labels")
c6.subheader(ytest.shape)
c6.table(ytest.head())

bag_clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=50,max_samples=10)
bag_clf.fit(xtrain,ytrain)

ypred=bag_clf.predict(xtest)

cr1=mat.classification_report(ytest,ypred,output_dict=True)

ac1=round(cr1['accuracy']*100,2)

m1=pickle.dump(bag_clf,open('bag.pkl','wb'))


rnd_clf=RandomForestClassifier(n_estimators=50,max_leaf_nodes=6,n_jobs=-1)

rnd_clf.fit(xtrain,ytrain)

ypredr=rnd_clf.predict(xtest)

cr2=mat.classification_report(ytest,ypredr,output_dict=True)

ac2=round(cr2['accuracy']*100,2)

m1=pickle.dump(rnd_clf,open('rnd.pkl','wb'))

c7,c8=st.columns(2)

c7.header("Accuracy of bagging classifier")
c7.subheader(ac1)

c8.header("Accuracy of random forest")
c8.subheader(ac2)