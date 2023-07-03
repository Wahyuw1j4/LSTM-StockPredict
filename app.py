from train_model import downloadData, ganerateData, initialData, collect_data
from manipJson import remove_zero_updated_data
from db import getStock
import time


if __name__ == "__main__":

    i = 0
    while True:
        tickerList = getStock()
        ticker = tickerList[i]
        data = downloadData(ticker)
        remove_zero_updated_data()
        if data.empty:
            print("data tidak ditemukan")
        else:
            index, jsonStock, predict_index= ganerateData(ticker, data)
            print(index)
            isUpdatedData = jsonStock['data'][index]['last_date'] != data.iloc[-1].name.strftime('%Y-%m-%d') # cek apakah data sudah up to date
            if isUpdatedData:
                x_train, y_train, train_len, scaled_data, valdataset, scaler, len_predict, dataset, lastDate = initialData(ticker, data)
                collect_data(ticker, jsonStock, index, data, dataset, lastDate, len_predict, train_len, scaled_data, scaler, valdataset, x_train, y_train, predict_index)
            else:
                print("data sudah up to date")
        i = i + 1
        if i == len(tickerList):
            print("sleeping for 1 hour")
            time.sleep(3600)
            i = 0
        time.sleep(5)
   