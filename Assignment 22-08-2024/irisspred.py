import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt



model = load_model('iris_model.keras')

s1=float(input("Enter the sepal length in cm: "))
s2=float(input("Enter the sepal width in cm: "))
s3=float(input("Enter the petal length in cm: "))
s4=float(input("Enter the petal width in cm: "))

sample = np.array([s1,s2,s3,s4]])

if st.