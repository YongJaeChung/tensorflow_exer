import tensorflow as tf
from tensorflow.keras import models
import pandas as pd
import matplotlib.pyplot as plt

samsung_df = pd.read_csv('005930.KS.csv')
print(samsung_df.head(5))

print(samsung_df.describe())

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
