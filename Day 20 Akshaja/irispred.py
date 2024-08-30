import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier as dtc
import streamlit as st

st.title(":tulip: Iris Machine Learning Model Prediction")

iris=pd.read_csv('Iris.csv')
x=iris.drop(columns=['Species'],axis=0)
y=iris['Species']
x.drop(columns=['Id'],axis=0,inplace=True)
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.25,random_state=10)

model=dtc(criterion='entropy')

model.fit(xtrain,ytrain)


st.subheader("Enter the size values to predict")


t1 =int(st.number_input("Enter sepal length"))
t2 =int(st.number_input("Enter sepal width"))
t3 =int(st.number_input("Enter petal length"))
t4 =int(st.number_input("Enter petal width"))

st.write("Sample Values : 6.2, 3.4, 5.4, 2.3")


sample1=[[t1,t2,t3,t4]]



if st.button("Predict Iris Species:"):
    target_sp=model.predict(sample1)
    st.write(target_sp)