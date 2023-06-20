from flask import Flask, request
import os
import csv,json
from flask_cors import CORS


app = Flask(__name__)
CORS(app, origins=['*'])


@app.route("/")
def hello_world():
    return "<h1>Hello, World!</h1>"

# UTILITIS

def csv_to_json(csv_path):
    # Read CSV file
    with open(csv_path, 'r') as csv_file:
        # Parse CSV data
        csv_data = csv.DictReader(csv_file)
        # Convert CSV to JSON
        json_data = json.dumps([row for row in csv_data])

    return json_data

@app.route("/allstock")
def getStockJson():
    filename = os.path.join(app.static_folder,'', '../../stock/stock.json')
    with open(filename) as test_file:
        data = json.load(test_file)
    return data

@app.route("/stock")
def getStockJsonByTicker():
    ticker = request.args.get("ticker")
    print(ticker)
    filename = os.path.join(app.static_folder,'..', '../stock/stock.json')
    with open(filename) as test_file:
        data = json.load(test_file)
    if ticker is None:
        print("masuk 1")
        return data
    else:
        print("masuk 2")
        for i, stock in enumerate(data['data']):
            if stock['name'] == ticker.upper():
                return stock


@app.route("/datastock")
def getStockCSV():
    ticker = request.args.get("ticker").upper()
    date = request.args.get("date")
    json_file_path = os.path.join(app.static_folder, '../../stock/stock.json')
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    if date is None:
        for i, stock in enumerate(data['data']):
            if stock['name'] == ticker:
                csv_file_path = os.path.join(app.static_folder, f"../../stock/{ticker}/{stock['last_date']}.csv")
                json_data = csv_to_json(csv_file_path)
                return json_data
    else:
        csv_file_path = os.path.join(app.static_folder, f"../../stock/{ticker}/{date}.csv")
        json_data = csv_to_json(csv_file_path)
        return json_data
    return "Data not found"
    
if __name__ == "__main__":
    app.run(debug=True)