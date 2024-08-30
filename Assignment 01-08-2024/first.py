import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
st.set_page_config(page_title="first page",page_icon=None,layout="centered",initial_sidebar_state="auto",menu_items=None)
st.write("Hello,*world!* :sunglasses:")

st.title("This is a title")
st.title("_python_ is :blue[cool]:clown_face:")


df1=pd.DataFrame({'column1':[1,2,3,4,5],'column2':[11,12,13,14,15]})
st.write("This is a DataFrame",df1)

st.header("one",divider=True)
st.markdown("#header1")
st.caption(":copyright: cdac trivandrum")
st.divider()

code='''def hello():
       print("hello world")'''
st.code(code,language="python")
with st.echo():
     st.write("this code will be printed")
     st.write(code)
     
## data elements
st.table(df1)
st.metric(label="temperature",value="35 *degree celsius",delta="1.2 *degree celsius")
st.json({"1":"one","numbers":[1,2,3,4,5]})


## charts

chart_data=pd.DataFrame(np.random.randn(20,3),columns=["a","b","c"])
st.area_chart(chart_data)
st.bar_chart(chart_data)
st.line_chart(chart_data)
st.scatter_chart(chart_data)

df2=pd.DataFrame({"Latitude":np.random.randn(1000)/50+37.76,"Longitude":np.random.randn(1000)/50+-122-4,"sizes":np.random.randn(1000)*100,"colors":np.random.randn(1000,4).tolist()})
st.map(df2,latitude='Latitude',longitude='Longitude',size='sizes',color='colors')


import plotly.express as px
df = px.data.gapminder().query("year == 2007").query("continent == 'Europe'")
df.loc[df['pop'] < 2.e6, 'country'] = 'Other countries' # Represent only large countries
fig = px.pie(df, values='pop', names='country', title='Population of European continent')