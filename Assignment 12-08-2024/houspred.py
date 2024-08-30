import streamlit as st
import pickle

st.set_page_config(page_title="Hitters",page_icon=":house_buildings:",layout="wide")

st.title(":house_buildings: Housing Data Analysis :house_buildings: ")

model1= pickle.load(open('rid1.pkl','rb'))
model2= pickle.load(open('lass1.pkl','rb'))
model3= pickle.load(open('enet1.pkl','rb'))

st.header("Prediction")
n1 = int(st.number_input("Enter value for area : "))
n2 = int(st.number_input("Enter value for bedrooms : "))
n3 = int(st.number_input("Enter value for bathrooms : "))
n4 = int(st.number_input("Enter value for stories : "))
n5 = int(st.number_input("Enter value for mainroad : "))
n6 = int(st.number_input("Enter value for guestroom : "))
n7 = int(st.number_input("Enter value for basement : "))
n8 = int(st.number_input("Enter value for hotwaterheating : "))
n9 = int(st.number_input("Enter value for airconditioning : "))
n10 = int(st.number_input("Enter value for parking : "))
n11 = int(st.number_input("Enter value for prefarea : "))
n12 = int(st.number_input("Enter value for furnishingstatus : "))


sample1 = [[n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12]]

if st.button("Predict the price"):
    t1=model1.predict(sample1)
    t2=model2.predict(sample1)
    t3=model3.predict(sample1)
    if (t1):
        st.write("Predicted price of the housing is:")
        c1,c2,c3 = st.columns(3)
        c1.subheader("Ridge Regression")
        c2.subheader("LASSO Regression")
        c3.subheader("ElasticNet Regression")
        c1.write(t1)
        c2.write(t2)
        c3.write(t3)
    else:
        st.write("price cannot be determined")