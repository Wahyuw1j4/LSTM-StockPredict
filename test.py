# import json



# with open('./stock/stock.json', 'r') as f:
#     data = json.load(f)

# for i, ticker in enumerate(data['data']):
#     if ticker['name'] == 'BBRI':
#         data['data'][i]['name'] = 'BBRI.JK'

# # Write the modified data back to the JSON file
# with open('./stock/stock.json', 'w') as f:
#     json.dump(data, f)


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


import time, math

print(math.floor(time.time()))