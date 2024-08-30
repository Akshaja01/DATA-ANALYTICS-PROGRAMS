import streamlit as st
import pickle


st.set_page_config(page_title="Diamond Data Analysis", page_icon="💎", layout="wide")
st.title("💍 Diamonds price Data Analysis 💍")


model1=pickle.load(open('gbr1.pkl','rb'))
model2=pickle.load(open('adr1.pkl','rb'))
#model3=pickle.load(open('xgb1.pkl','rb'))
#model4=pickle.load(open('cat.pkl','rb'))

c1,c2=st.columns(2)

n1=c1.number_input("carat")
n2=c2.number_input("cut")
n3=c1.number_input("color")
n4=c2.number_input("clarity")
n5=c1.number_input("depth")
n6=c2.number_input("table")
n7=c1.number_input("x")
n8=c2.number_input("y")
n9=c2.number_input("z")

new_features=[[n1,n2,n3,n4,n5,n6,n7,n8,n9]]

cl1,cl2,cl3,cl4=st.columns(4)

if cl1.button("model1 prediction"):
    t1 = model1.predict(new_features)
    cl1.subheader(t1)
   
if cl2.button("model2 prediction"):
    t2 = model2.predict(new_features)
    cl2.subheader(t2)

#if cl3.button("model3 prediction"):
    #t3 = model3.predict(new_features)
    #cl3.subheader(t3)
