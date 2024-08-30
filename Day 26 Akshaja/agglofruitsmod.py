import streamlit as st
import pickle
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(page_title="Agglomerative clustering in fruits data", page_icon="🍒", layout="wide")
st.title("🍓 Agglomerative clustering in fruits data 🍓️")

model1=pickle.load(open('agg1.pkl','rb'))
model2=pickle.load(open('agg2.pkl','rb'))
model3=pickle.load(open('agg3.pkl','rb'))

st.header("Prediction")
c1, c2 = st.columns(2)
n1=int(c1.number_input("Enter mass value"))
n2=int(c1.number_input("Enter width"))
n3=int(c2.number_input("Enter height"))
n4=int(c2.number_input("Enter color score"))

# st.write("Sample values: mass, width, height, color_score")
# st.write("Sample values: apple : 180, 8, 6.8, 0.59")
# st.write("Sample values: mandarin: 76, 5.8, 4, 0.81")
# st.write("Sample values: orange : 154, 7.2, 7.2, 0.82")
# st.write("Sample values: lemon : 118, 6.1, 8.1, 0.7")

sample=[[n1,n2,n3,n4]]

c1,c2,c3=st.columns(3)

if(c1.button("Predict the Fruit Label for single linkage")):
    t = model1.predict(sample)
    if (t == 0):
        c1.write(":apple: Apple")
    elif (t == 1):
        c1.write(":tangerine: Mandarin")
    elif (t == 2):
        c1.write(":tangerine: Orange")
    elif (t == 3):
       c1.write(":lemon: Lemon")        
    else:
        c1.write("Fruit not listed")


if(c2.button("Predict the Fruit Label for complete linkage")):
    t = model2.predict(sample)
    if (t == 0):
        c2.write(":apple: Apple")
    elif (t == 1):
        c2.write(":tangerine: Mandarin")
    elif (t == 2):
        c2.write(":tangerine: Orange")
    elif (t == 3):
       c2.write(":lemon: Lemon")        
    else:
        c2.write("Fruit not listed")       

if(c3.button("Predict the Fruit Label for average linkage")):
    t = model3.predict(sample)
    if (t == 0):
        c3.write(":apple: Apple")
    elif (t == 1):
        c3.write(":tangerine: Mandarin")
    elif (t == 2):
        c3.write(":tangerine: Orange")
    elif (t == 3):
       c3.write(":lemon: Lemon")        
    else:
        c3.write("Fruit not listed")                 