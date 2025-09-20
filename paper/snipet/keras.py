import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# Nạp bộ dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
model = Sequential()
model.add(Convolution2D(32,3,3,activation='relu', input_shape=(1,28,28)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, nb_epoch=10, verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)
