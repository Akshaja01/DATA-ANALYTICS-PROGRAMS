import streamlit as st
import pickle

st.set_page_config(page_title="Diamond Data Analysis", page_icon="💎", layout="wide")
st.title("💎 Diamond Data Analysis 💎")
model1=pickle.load(open('gbr1.pkl','rb'))
model2=pickle.load(open('adr1.pkl','rb'))
model3=pickle.load(open('xgb1.pkl','rb'))
model4=pickle.load(open('cat1.pkl','rb'))

#  
c11,cl2,cl3,cl4=st.columns(4)
n1=c1.number_input("carat")
n2=c2.number_input(" cut")
n3=c1.number_input("color")
n4=c2.number_input("clarity")
n5=c1.number_input("depth")
n6=c2.number_input("table")
n7=c1.number_input("price")
n8=c2.number_input("Age")
n9=

new_features=[[n1,n2,n3,n4,n5,n6,n7,n8]]

if st.button("Predict"):
    c5,c6=st.columns(2)
    t1=model1.predict(new_features)
    c5.subheader("Result 1 ")
    if t1==1:
        c5.write("Has diabetes")
    elif t1==0:
        c5.write("Does not have diabetes")
    else:
        c5.write("Cannot predict")
   
    t1=model2.predict(new_features)
    c6.subheader("Result 2 ")
    if t1==1:
        c6.write("Has diabetes")
    elif t1==0:
        c6.write("Does not have diabetes")
    else:
        c6.write("Cannot predict")
        
        cut	color	clarity	depth	table	price	x	y	z
