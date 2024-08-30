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
from tensorflow.keras.callbacks import ModelCheckpoint


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

checkpoint = ModelCheckpoint("iris_model.keras", monitor='accuracy', verbose=1, save_best_only=True, mode='auto')
history = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=50, batch_size=10, callbacks=[checkpoint])
hdf = pd.DataFrame(history.history)
print("Data Frame")
print(hdf)



plt.figure(figsize=(10, 5))
plt.plot(hdf['accuracy'], label='Training Accuracy')
plt.plot(hdf['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(hdf['loss'], label='Training Loss')
plt.plot(hdf['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()





n1 = float(input("Enter sepal length:"))
n2 = float(input("Enter sepal width:"))
n3 = float(input("Enter petal length:"))
n4 = float(input("Enter petal width:"))

pred = model.predict(np.array([[n1, n2, n3, n4]]))
pred = np.argmax(pred, axis=1)


if (pred==0):
    print("Setosa")
elif (pred==1):
    print("Virginica")
elif (pred==2):
    print("Versicolor")
else:
    print("Flower not listed")