import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import datetime
# import minmaxscaler

import json

ticker = 'BBRI.JK'


# json read
with open('./stock/stock.json', 'r') as f:
    jsonStock = json.load(f)

#  get index of ticker
index = None
for i, stock in enumerate(jsonStock['data']):
    if stock['name'] == ticker:
        index = i
        break

# get last date
lastDate = jsonStock['data'][index]['last_date']

csv = pd.read_csv(f'stock/{ticker}/{lastDate}.csv')

data = csv.filter(['Close'])
x_test = data.values[-60:]
print(x_test)
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(x_test)

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))

# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
model = tf.keras.models.load_model(f'stock/{ticker}/model_{ticker}.h5')
predict = model.predict(x_test)
predict = scaler.inverse_transform(predict)
print(predict)