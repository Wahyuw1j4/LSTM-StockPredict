import json


def remove_zero_updated_data():
    with open('./stock/stock.json', 'r') as f:
        jsonStock = json.load(f)

    # Menghapus data dengan updated_at = 0
    for item in jsonStock['data']:
        item['data_predict'] = [d for d in item['data_predict'] if d['updated_at'] != 0]

    # Mengembalikan data JSON yang telah dihapus
    with open('./stock/stock.json', 'w') as f:
        json.dump(jsonStock, f)



