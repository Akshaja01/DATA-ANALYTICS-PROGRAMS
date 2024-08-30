import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split as tts
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.datasets import load_iris
import numpy as np


iris = load_iris()
x = iris.data
y = iris.target

print("Features of Iris dataset:")
print(x)

print("\nLabels of Iris dataset:")
print(y)


xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2, random_state=10, shuffle=True, stratify=y)


ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)


print("\nOne-hot encoded labels of training data:")
print(ytrain)

print("\nOne-hot encoded labels of testing data:")
print(ytest)

model = Sequential()

model.add(Dense(10, input_shape=(4,), activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


print("\nModel Summary:")
model.summary()


history = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=50, batch_size=10)


plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()