import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split as tts
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint


df = pd.read_csv("pizza_price-1.csv")
x = df.drop(columns=["Restaurant","Price"])
y = df[["Price"]]



print("Features of Pizza dataset:")
print(x)

print("\nLabels of Pizza dataset:")
print(y)


model = Sequential()


model.add(Dense(128, input_dim=4, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))


model.compile(optimizer="Adam", loss='mae', metrics=['mae'])

print("\nModel Summary:")
model.summary()

#checkpoint = ModelCheckpoint("pizza_model.keras", monitor='accuracy', verbose=1, save_best_only=True)

history = model.fit(x, y, epochs=100, batch_size=8, verbose=2)

hdf = pd.DataFrame(history.history)
print("Data Frame")
print(hdf)


model.save("pizza_model.keras")



plt.figure(figsize=(10, 5))
plt.plot(hdf['loss'], label='loss')

plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Epochs and Loss')
plt.legend()
plt.show()



print("Prediction")
n1=int(input("Extra_Cheeze:"))
n2=int(input("Extra_Mushroom:"))
n3=int(input("Size_Inch:"))
n4=int(input("Extra_Spicy:"))


pred = model.predict(np.array([[n1, n2, n3, n4]]))
print("Predicted Price of the Pizza is:")
print(pred)