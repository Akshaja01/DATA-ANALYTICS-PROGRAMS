

import streamlit as st
from spam_eda_models import model1, model2, count_vect

cv = count_vect()
mnb = model1()
bnb = model2()

t1 = st.text_input("Enter a text")
st.write("Sample Text for spam: Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's")

st.write(" You Entered :", t1)
t1 = [t1]
sample1 = cv.transform(t1)


if st.button("Predict Spam or not spam"):
   
    target_label1 = mnb.predict(sample1)
    target_label2 = bnb.predict(sample1)
   
    if (target_label1 == 1):
        target_label1_text = "Spam"
    else:
        target_label1_text = "Not Spam"
    
    if (target_label2 == 1):
        target_label2_text = "Spam"
    else:
        target_label2_text = "Not Spam"
    c1, c2 = st.columns(2)
    c1.subheader("Prediction with MultinomialNB")
    c1.write(target_label1_text)
       
    
    c2.subheader("Prediction with BernoulliNB")
    c2.write(target_label2_text)
    
    


