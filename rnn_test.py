import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import Sequential

def seq2dataset(seq,window,horizon):
    X = []
    Y = []

    for i in range(len(seq)-(window+horizon)+1):
        x = seq[i:(i+window)]
        y = (seq[i+window+horizon-1])

        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)



x = np.arange(0,100,0.1)
y = 0.5*np.sin(2*x) - np.cos(x/2.0)
seq_data = y.reshape(-1,1)

print(seq_data.shape)
print(seq_data[:5])
'''
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False
plt.grid()
plt.title('0.5*sin(2x)-cos(x/2)')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.plot(seq_data)
plt.show()
'''

w = 20
h = 1

X,Y = seq2dataset(seq_data,w,h)

print(X.shape , Y.shape)

split_ratio = 0.8

split = int(split_ratio*len(X))

x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]


print(x_train.shape , y_train.shape,
      x_test.shape, y_test.shape)


model = Sequential()
model.add(SimpleRNN(units=128,
                    activation='tanh',
                    input_shape=x_train[0].shape))
model.add(Dense(1))
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mae'])

hist = model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))
