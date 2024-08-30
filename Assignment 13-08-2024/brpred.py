import streamlit as st
import pickle

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="️:hospital:", layout="wide")
st.title("🎗️ Breast Cancer Prediction 🎗️")


model1 = pickle.load(open('breast_svcm1.pkl', 'rb'))
model2 = pickle.load(open('breast_polysvc.pkl', 'rb'))
model3 = pickle.load(open('breast_polykernel.pkl', 'rb'))
model4 = pickle.load(open('breast_rbfkernel.pkl', 'rb'))
model5 = pickle.load(open('breast_rbfc1f.pkl', 'rb'))


c1, c2 = st.columns(2)
n1 = c1.number_input("Mean Radius")
n2 = c2.number_input("Mean Texture")
n3 = c1.number_input("Mean Perimeter")
n4 = c2.number_input("Mean Area")
n5 = c1.number_input("Mean Smoothness")
n6 = c2.number_input("Mean Compactness")
n7 = c1.number_input("Mean Concavity")
n8 = c2.number_input("Mean Concave Points")
n9 = c1.number_input("Mean Symmetry")
n10 = c2.number_input("Mean Fractal Dimension")

new_features = [[n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]]


c3, c4, c5, c6, c7 = st.columns(5)

if c3.button("Model 1 Prediction"):
    t1 = model1.predict(new_features)
    c3.subheader("Prediction")
    if t1 == 0:
        st.write("Benign")
    else:
        st.write("Malignant")

if c4.button("Model 2 Prediction"):
    t2 = model2.predict(new_features)
    c4.subheader("Prediction")
    if t2 == 0:
        st.write("Benign")
    else:
        st.write("Malignant")

if c5.button("Model 3 Prediction"):
    t3 = model3.predict(new_features)
    c5.subheader("Prediction")
    if t3 == 0:
        st.write("Benign")
    else:
        st.write("Malignant")

if c6.button("Model 4 Prediction"):
    t4 = model4.predict(new_features)
    c6.subheader("Prediction")
    if t4 == 0:
        st.write("Benign")
    else:
        st.write("Malignant")

if c7.button("Model 5 Prediction"):
    t5 = model5.predict(new_features)
    c7.subheader("Prediction")
    if t5 == 0:
        st.write("Benign")
    else:
        st.write("Malignant")