import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import datetime
# import minmaxscaler

import json

ticker = 'BBRI.JK'
today = datetime.date.today()
csv = pd.read_csv(f'stock/{ticker}/{today}.csv')

 
data = csv.filter(['Close'])
x_test = data.values[-60:]
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(x_test)

x_test = np.array(x_test)
x_test = np.reshape(x_test, (1, 60, 1))

# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
model = tf.keras.models.load_model(f'stock/{ticker}/model_{ticker}.h5')
predic = model.predict(x_test)
predic = scaler.inverse_transform(predic)
print(predic)