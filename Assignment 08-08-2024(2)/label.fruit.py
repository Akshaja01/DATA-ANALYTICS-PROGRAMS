import streamlit as st
import numpy as np
from sklearn import metrics as mat
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier
from fruit import data
from sklearn.model_selection import GridSearchCV as gscv


st.title(":grapes: fruit label prediction")
x,y=data()

xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.2,random_state=0)

c1,c2=st.columns(2)

c1.subheader('training data size')
c1.write(xtrain.shape)
c1.write(ytrain.shape)

c2.subheader('testing data size')
c2.write(xtest.shape)
c2.write(ytest.shape)


knn1=KNeighborsClassifier()
param={'n_neighbors':np.arange(1,10)}
 
knn_gscv=gscv(knn1,param,cv=5)


knn_gscv.fit(xtrain,ytrain)



c3,c4=st.columns(2)

c3.subheader('best neighbors')


c3.write(knn_gscv.best_params_['n_neighbors'])

 #c4.subheader('best_score')

#c4.write(knn_gscv.best_score)



#knnmodel=KNeighborsClassifier(n_neighbors=n1)

#knnmodel.fit(xtrain,ytrain)
#ypred=knnmodel.predict(xtest)

st.header("prediction")
n1=int(st.number.input("enter mass"))
n2=int(st.number.input("enter width"))
n3=int(st.number.input("enter height"))
n4=int(st.number.input("enter color_score"))

sample=[[n1,n2,n3,n4]]

if(st.button



