from mysql.connector import (connection)

def getStock():
    cnx = connection.MySQLConnection(user='root', password='',
                                 host='127.0.0.1',
                                 database='stock_predict')
    cursor = cnx.cursor()
    cursor.execute("SELECT * FROM stock")
    cursor = cursor.fetchall()
    stock = []
    for x in cursor:
        stock.append(x[1])
    cnx.close()

    return stock

