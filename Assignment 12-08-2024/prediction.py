import streamlit as st
import pickle

st.set_page_config(page_title="Hitters",page_icon=":money_with_wings:",layout="wide")

st.title(":money_with_wings: Hitter Data Analysis :money_with_wings: ")

model1= pickle.load(open('rid1.pkl','rb'))
model2= pickle.load(open('lass1.pkl','rb'))
model3= pickle.load(open('enet1.pkl','rb'))

st.header("Prediction")
n1 = int(st.number_input("Enter value for AtBat : "))
n2 = int(st.number_input("Enter value for Hits : "))
n3 = int(st.number_input("Enter value for HmRun : "))
n4 = int(st.number_input("Enter value for Runs : "))
n5 = int(st.number_input("Enter value for RBI : "))
n6 = int(st.number_input("Enter value for Walks : "))
n7 = int(st.number_input("Enter value for Years : "))
n8 = int(st.number_input("Enter value for CAtBat : "))
n9 = int(st.number_input("Enter value for CHits : "))
n10 = int(st.number_input("Enter value for CHmRun : "))
n11 = int(st.number_input("Enter value for CRuns : "))
n12 = int(st.number_input("Enter value for CRBI : "))
n13 = int(st.number_input("Enter value for CWalks : "))
n14 = int(st.number_input("Enter value for League : "))
n15 = int(st.number_input("Enter value for Division : "))
n16 = int(st.number_input("Enter value for PutOuts : "))
n17 = int(st.number_input("Enter value for Assists : "))
n18 = int(st.number_input("Enter value for Errors : "))
n19 = int(st.number_input("Enter value for NewLeague : "))

sample1 = [[n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19]]

if st.button("Predict the salary"):
    t1=model1.predict(sample1)
    t2=model2.predict(sample1)
    t3=model3.predict(sample1)
    if (t1):
        st.write("Predicted salary of the hitter is:")
        c1,c2,c3 = st.columns(3)
        c1.subheader("Ridge Regression")
        c2.subheader("LASSO Regression")
        c3.subheader("ElasticNet Regression")
        c1.write(t1)
        c2.write(t2)
        c3.write(t3)
    else:
        st.write("salary cannot be determined")