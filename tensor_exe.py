import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import SGD,Adam,RMSprop

import sys

try:
    #loaded_data = np.loadtxt('diabetes.csv',delimiter=',')

    loaded_data = pd.read_csv('diabetes.csv')
    #print(loaded_data)


    x_data = loaded_data.iloc[:,0:-1].values
    t_data = loaded_data.iloc[:,-1].values
    #t_data = t_data.reshape(-1,1)
    t_data = t_data.reshape(-1,1)
    print(x_data)
    #print(t_data.head(5))
    print(x_data.shape)
    print(t_data.shape)


except Exception as err:
    print(err)


model = models.Sequential()

model.add(Dense(t_data.shape[1],input_shape=(x_data.shape[1],),activation='sigmoid'))

model.compile(optimizer=SGD(learning_rate=0.01),loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

# hist.history['loss'] , hist.history['val_loss'] hist.history['accuracy'] , hist.history['val_accuracy']
hist = model.fit(x_data, t_data, epochs=500, validation_split=0.2, verbose=2)

model.evaluate(x_data,t_data)


import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'

'''
plt.title('손실 그래프')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.plot(hist.history['loss'],label='train_loss')
plt.plot(hist.history['val_loss'],label='validation_loss')
plt.legend(loc='best')
plt.show()
'''

plt.title('정확도 그래프')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()
plt.plot(hist.history['accuracy'],label='train_accuracy')
plt.plot(hist.history['val_accuracy'],label='validation_accuracy')

plt.legend(loc='best')
plt.show()
