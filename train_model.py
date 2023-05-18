import yfinance as yf
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import time
import os
import json
import pandas as pd

# plt.style.use('fivethirtyeight')
def downloadData(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2024-04-30")
    return data

def ganerateData(ticker):
    with open('./stock/stock.json', 'r') as f:
        jsonStock = json.load(f)

    index = None
    predict_index = 0
    isTickerExist = False

    for i, stock in enumerate(jsonStock['data']):
        if stock['name'] == ticker:
            index = i
            isTickerExist = True
            jsonStock['data'][index]['data_predict'].append({
                "updated_at": 0,
                "name_file": "",
                "rmse": 0,
                "predict": 0,
            })
            predict_index = len(jsonStock['data'][index]['data_predict']) - 1
            with open('./stock/stock.json', 'w') as f:
                json.dump(jsonStock, f)
            break
        else:
            isTickerExist = False

    if isTickerExist == False:
        jsonStock['data'].append({
            "name": ticker,
            "last_date": "",
            "data_predict": [{
                "updated_at": 0,
                "name_file": "",
                "rmse": 0,
                "predict": 0,
            }],
        })
        
        index = len(jsonStock['data']) - 1
        predict_index = 0 
        with open('./stock/stock.json', 'w') as f:
            json.dump(jsonStock, f)
        return index, jsonStock, predict_index
    else:
        return index, jsonStock, predict_index

def initialData(ticker, data):
    ts = data.iloc[-1].name
    lastDate = ts.strftime('%Y-%m-%d')
    dataset = data.filter(['Close'])
    if not os.path.exists(f'stock/{ticker}'):
        os.makedirs(f'stock/{ticker}')
    valdataset = dataset.values
    train_len = math.ceil(len(valdataset) * 0.8)
    len_predict = len(valdataset) - train_len
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(valdataset)
    train_data = scaled_data[0:train_len]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    return x_train, y_train, train_len, scaled_data, valdataset, scaler, len_predict, dataset,lastDate

def trainData(ticker, x_train, y_train, train_len, scaled_data, valdataset, scaler):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True,
              input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1, callbacks=[EarlyStopping(monitor='loss', patience=5)])
    
    model.save(f'stock/{ticker}/Model_{ticker}.h5')

    test_data = scaled_data[train_len - 60:, :]
    x_test = []
    y_test = valdataset[train_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predict = model.predict(x_test)
    predict = scaler.inverse_transform(predict)

    return predict, y_test, model


def collect_data(ticker, jsonStock, index, data, dataset, lastDate, len_predict, train_len, scaled_data, scaler, valdataset, x_train, y_train, predict_index):
    rmse = 9990
    while rmse > 300:
        predict, y_test, model = trainData(ticker, x_train, y_train, train_len, scaled_data, valdataset, scaler)
        rmse = np.sqrt(np.mean(predict - y_test)**2)
        print(rmse)

    jsonStock['data'][index]["data_predict"][predict_index]['rmse'] = rmse.item()
    jsonStock['data'][index]["data_predict"][predict_index]['updated_at'] = math.floor(time.time())
    jsonStock['data'][index]["data_predict"][predict_index]['name_file'] = f'{lastDate}.csv'
    jsonStock['data'][index]['last_date'] = lastDate

    dataset['predict'] = 0

    dataset.loc[data.index[-len_predict:], 'predict'] = predict
    dataset.to_csv(f'stock/{ticker}/{lastDate}.csv')

    predict_data = scaled_data[-60:]
    predict_data = np.array(predict_data)
    prediction = model.predict(predict_data.reshape(1, 60, 1))
    prediction = scaler.inverse_transform(prediction)
    jsonStock['data'][index]["data_predict"][predict_index]['predict'] = prediction.item()
    print(prediction.item())

    with open('./stock/stock.json', 'w') as f:
        json.dump(jsonStock, f)




# print(valid)
# plt.figure()
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'predict']])
# plt.legend(['train', 'val', 'predict'])
# plt.show()
