import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier as dtc
import streamlit as st
from sklearn import metrics as mat

st.title(":tulip: Iris Machine Learning Model Prediction")

iris=pd.read_csv('Iris.csv')
st.dataframe(iris.head(5))
x=iris.drop(columns=['Species'],axis=0)
y=iris['Species']
x.drop(columns=['Id'],axis=0,inplace=True)
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.25,random_state=10)
model=dtc(criterion='entropy')
model.fit(xtrain,ytrain)
st.header("performance measures of iris model")

accuracy_iris=mat.accuracy_score(ytest,ypred)

precision_iris=mat.precision_score(ytest,ypred)

col1.subheader("Accuracy score for iris model")

col1.subheader(accuracy_iris)

col2.subheader("Precision score for iris model")

col2.subheader(precision_iris)

st.subheader("Confusion_matrix")

cm=mat.confusion_matrix(ytest,ypred)

fig=px.imshow(cm,text_auto=True)
st.plotly_chart(fig,use_container_width=True)

st.subheader("Classification report")
st.subheader(mat.classification_report(ytest,ypred))
st.header("Prediction on Iris model")
