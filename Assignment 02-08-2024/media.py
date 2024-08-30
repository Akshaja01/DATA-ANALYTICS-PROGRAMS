import streamlit as st
import pandas as pd
genre=st.radio("what is your favourite movie game",["comedy","drama","documentery"],
captions=["laughout loud","get the popcorn","never stop"])
if genre=="comedy":
    st.write("you have selected comedy")
else:
    st.write("you didint")
    
    
## select box

option=st.selectbox("how would yoyu like to be connected",("email","hornp=hone","mobilepjone","whatsapp"))
st.write("you have selected",option)

## select slider

color=st.select_slider("select color of rainbbow",options=["red","green","blue","indigo","violet","yellow"])
st.write("the color you have selected is",color)

## toggle

on=st.toggle("Activate your connection")
if on:
     st.write("connection",on)

st.image("cl.jpg",caption="cloud")

video1=open("rabbit.mp4","rb")
video1_bytes=video1.read()
st.video(video1_bytes)

audio1=open("melody.mp3","rb")
audio1_bytes=audio1.read()
st.audio(audio1_bytes,format="audio/mp3")

col1,col2,col3=st.columns(3)
with col1:
    st.header("figure1")
    st.image("1.jpg")
with col2:
    st.header("figure2")
    st.image("2.jpg")
with col3:
    st.header("figure3")
    st.image("3.jpg")
    
    
cont1=st.container(border=True)
cont1.write("this is inside the container")
st.write("this is  outside the container")


st.bar_chart({"data":[1,5,2,6,2,1]})
with st.expander("see explantaion"):
    st.write('''abcdefgh''')
    
    
prompt=st.chat_input("say something")
if prompt:
    st.write(prompt)
    
msg=st.chat_message("assitant")
msg.write("hello")
