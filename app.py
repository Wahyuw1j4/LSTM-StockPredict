import yfinance as yf
import numpy as np
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
plt.style.use('fivethirtyeight')


today = datetime.date.today()

ticker = "BBRI.JK"
data = yf.download(ticker, start="2010-01-01", end="2023-04-30")
dataset = data.filter(['Close'])
dataset.to_csv(f'stock/{ticker}/{today}.csv')
valdataset = dataset.values


train_len = math.ceil(len(valdataset) * 0.8)

scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(valdataset)

train_data = scaled_data[0:train_len]
x_train = []
y_train = []
for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])

# print(x_train)

x_train,y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
x_train.shape

callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]

# def train(): 
#   model = Sequential()
#   model.add(LSTM(32, return_sequences=True, input_shape = (x_train.shape[1], 1)))
#   model.add(LSTM(64, return_sequences=False))
#   model.add(Dense(25))
#   model.add(Dense(1))

#   model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#   model.fit(x_train, y_train, validation_split=0.2, batch_size=64, epochs=30, callbacks=callback)

#   test_data = scaled_data[train_len - 60:, :]
#   x_test = []
#   y_test = valdataset[train_len:,:]
#   for i in range(60, len(test_data)):
#     x_test.append(test_data[i-60:i,0])

#   x_test = np.array(x_test)
#   x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
#   predic = model.predict(x_test)
#   predic = scaler.inverse_transform(predic)

#   return predic, y_test

def train(): 
  model = Sequential()
  model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
  model.add(LSTM(64, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))

  model.compile(optimizer='adam', loss='mean_squared_error')
  model.fit(x_train, y_train, batch_size=1, epochs=1)

  test_data = scaled_data[train_len - 60:, :]
  x_test = []
  y_test = valdataset[train_len:,:]
  for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
  predic = model.predict(x_test)
  predic = scaler.inverse_transform(predic)

  return predic, y_test


rmse = 999
while rmse > 10:
  predic, y_test = train()
  rmse = np.sqrt(np.mean(predic - y_test)**2)
  print(rmse)

train = data[:train_len]
valid = data[train_len:]
valid['predic'] = predic
print(valid)
plt.figure()
plt.plot(train['Close'])
plt.plot(valid[['Close', 'predic']])
plt.legend(['train', 'val', 'predic'])
plt.show()