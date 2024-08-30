import streamlit as st
import numpy as np
model=load_model'iris_model.keras')
st.title("Irisi Data Analysis")
st.divider()

st.header("Prediction of iris species")  

s1=st.number_input("Enter the sepal length cm")
s2=st.number_input("Enter the sepal width cm")
s3=st.number_input("Enter the petal length cm")
s4=st.number_input("Enter the petal width cm")

sample1=np.array([[s1,s2,s3,s4]])
if st.button("Predict the species"):
    target_sp=moderl.predict(sample1)
    st.write(target_sp)
    out=np.argmax(target_sp,axis=1)
if out=0:
    st.subheader('Iris setosa')
    st.image('setosa.jpg')
elif out=1:
    st.subheader('Iris virginica')
    st.image('virginica.jpg')
elif out=2:
    st.subheader('Iris versicolor')
    st.image('versicolor.jpg')
else:
    st.subheader('Flower not listed')