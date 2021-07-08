import tensorflow as tf
from tensorflow.keras import models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

samsung_df = pd.read_csv('005930.KS.csv')
print(samsung_df.head(5))

print(samsung_df.describe())
'''
samsung_df.hist()
plt.show()
plt.figure(figsize=(7,4))
plt.title('Samsung elec stock price')
plt.xlabel('price')
plt.ylabel('period')
plt.grid()
plt.plot(samsung_df['Adj Close'], label='Adj Close',color='b')
plt.legend(loc='best')
plt.show()
'''
samsung_df['Volume'] = samsung_df['Volume'].replace(0,np.nan)
print(samsung_df.isnull().sum())
samsung_df['3MA'] = 0
samsung_df['3MA'].loc[2:] = samsung_df['Adj Close'].loc[samsung_df['Adj Close'].index - 2] + samsung_df['Adj Close'].loc[samsung_df['Adj Close'].index - 1]

samsung_df
scaler = MinMaxScaler()
scale_cols = ['']
