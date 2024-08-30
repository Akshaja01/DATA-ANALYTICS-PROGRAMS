import streamlit as st
import pickle

st.set_page_config(page_title="Iris Species prediction",page_icon=" :tulip: ",layout="wide")
st.title(" :tulip: iris prediction for species :tulip: ")

model1=pickle.load(open('svcm1.pkl','rb'))
model2=pickle.load(open('polysvc.pkl','rb'))
model3=pickle.load(open('polykernel.pkl','rb'))
model4=pickle.load(open('rbfkernel.pkl','rb'))
model5=pickle.load(open('rbfc1f.pkl','rb'))

c1,c2=st.columns(2)
n1=c1.number_input("Sepal length")
n2=c2.number_input("Sepal width")
n3=c1.number_input("petal length")
n4=c2.number_input("petal width")
c3,c4,c5,c6,c7=st.columns(5)
new_features=[[n1,n2,n3,n4]]

if c3.button("model 1 prediction"):
    t1=model1.predict(new_features)
    c3.subheader("Predicted species is")
    if t1==0:
        st.write("Iris setosa")
    elif t1==1:
        st.write("Iris versicolor")
    elif t1==2:
        st.write("Iris virginica")
    else:
        st.write("flower not listed")

if c4.button("model 2 prediction"):
    t2=model2.predict(new_features)
    c4.subheader("Predicted species is")
    if t2==0:
        st.write("Iris setosa")
    elif t2==1:
        st.write("Iris versicolor")
    elif t2==2:
        st.write("Iris virginica")
    else:
        st.write("flower not listed")        

if c5.button("model 3 prediction"):
    t3=model3.predict(new_features)
    c5.subheader("Predicted species is")
    if t3==0:
        st.write("Iris setosa")
    elif t3==1:
        st.write("Iris versicolor")
    elif t3==2:
        st.write("Iris virginica")
    else:
        st.write("flower not listed")

if c6.button("model 4 prediction"):
    t4=model1.predict(new_features)
    c3.subheader("Predicted species is")
    if t4==0:
        st.write("Iris setosa")
    elif t4==1:
        st.write("Iris versicolor")
    elif t4==2:
        st.write("Iris virginica")
    else:
        st.write("flower not listed")

if c7.button("model 5 prediction"):
    t5=model1.predict(new_features)
    c3.subheader("Predicted species is")
    if t5==0:
        st.write("Iris setosa")
    elif t5==1:
        st.write("Iris versicolor")
    elif t5==2:
        st.write("Iris virginica")
    else:
        st.write("flower not listed")        