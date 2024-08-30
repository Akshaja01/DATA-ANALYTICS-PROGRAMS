import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split as tts
from keras.layers import Dense
from keras.models import sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.datasets import load_iris
from tenserflow.keras.callbacks import checkpoints
import numpy as np



st.set_page_config(page_title="iris Data Analysis", page_icon="🌸", layout="wide")
st.title("🌸 iris Data Analysis 🌸")
st.divider()

iris=load_iris()
x=iris.data
y=iris.target
c1,c2=st.columns(2)
c1.header("featutures of iris")
c1.table(x)

c2.header("label of iris")
c2.table(y)


xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2, random_state = 10,shuffle=True,stratify=y)
ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)
c1,c2=st.columns(2)
c1.header("labels of training")
c1.table(ytrain)

c2.header("labels of testing")
c2.table(ytest)

model=sequential()

model.add(Dense(10,input_shape=(4),activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(3,activation="softmax"))
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
st.header("model summary")
st.write(model.summary())