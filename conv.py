import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten
from tensorflow.keras.optimizers import Adam
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(-1,28,28,1) ## 높이 너비 채널
x_test = x_test.reshape(-1,28,28,1)

print(x_train.shape , x_test.shape)
print(y_train.shape , y_test.shape)

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

cnn = models.Sequential()

cnn.add(Conv2D(input_shape=(28,28,1),
                 kernel_size=(3,3),
                 filters=32,
                 strides=(1,1),
                 activation='relu',
                 use_bias=True,
                 padding='valid'
                )
         )

cnn.add(Conv2D(kernel_size=(3,3),
                 filters=32,
                 strides=(1,1),
                 activation='relu',
                 use_bias=True,
                 padding='valid'
                )
         )
cnn.add(MaxPool2D(pool_size=(2,2),padding='valid'))

cnn.add(Dropout(0.25))

cnn.add(Flatten())

cnn.add(Dense(128,activation='relu'))

cnn.add(Dropout(0.5))

cnn.add(Dense(10,activation='softmax'))

cnn.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

hist = cnn.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test))

cnn.evaluate(x_test,y_test)
