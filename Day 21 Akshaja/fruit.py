import streamlit as st
import numpy as np
from sklearn import metrics as mat
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier
from fruitsnew import data
from sklearn.model_selection import GridSearchCV as gscv
import plotly.express as px


st.title(":grapes: fruit label predicyion")
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

n_1=knn_gscv.best_params_['n_neighbors']
c3.write(n_1)

c4.subheader('best_score')

c4.write(knn_gscv.best_score_*100)



knnmodel=KNeighborsClassifier(n_neighbors=n_1)

knnmodel.fit(xtrain,ytrain)
ypred=knnmodel.predict(xtest)

st.subheader("classification report")
st.table(mat.classification_report(ytest,ypred,output_dict=True))

st.subheader("confusion matrix of KNN")

cm1=mat.confusion_matrix(ytest,ypred,labels=[1,2,3,4])

fig1=px.imshow(cm1,text_auto=True,labels=dict(x="predicted values",y="actual values"),
x=['Apple','Mandarin','Orange','Lemon'],
y=['Apple','Mandarin','Orange','Lemon'])
st.plotly_chart(fig1)

st.header("prediction")
n1=int(st.number_input("enter mass"))
n2=int(st.number_input("enter width"))
n3=int(st.number_input("enter height"))
n4=int(st.number_input("enter color_score"))

sample=[[n1,n2,n3,n4]]

if st.button("Predict the fruit label"):
    t=knnmodel.predict(sample)
    if t==1:
        st.write("Apple")
    elif t==2:
        st.write("Mandarin")
    elif t==3:
        st.write("Orenge")
    elif t==3:
        st.write("melon")
    else:
        pass