import streamlit as st
import pickle
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="Salary",page_icon=" :moneybag: ",layout="wide")
st.title(" :money_with_wings: Salary Prediction :moneybag: ")

salary_model = pickle.load(open('salary.pkl', 'rb'))

st.header("Salary Prediction")
n1=int(st.number_input("Enter Employee level (1,2,3,4,5): "))

sample1=[[n1]]
poly   = PolynomialFeatures(degree=4)
xpoly  = poly.fit_transform(sample1)

if st.button("Predict salary of the Employee"):
    t= salary_model.predict(xpoly)
    if (t):
        st.write("Predicted Salary is:")
        st.write(t)
    else :
          st.write("Salary cannot be predicted.")