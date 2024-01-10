#==================== IMPORT PACKAGES =============================

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


#=================== READ A INPUT DATA ============================

dataframe=pd.read_csv("creditcard_1.csv")
print("----------------------------------------------------------")
print("Data Selection")
print()
print(dataframe.head(10))
print()
print("-----------------------------------------------------------")

list(dataframe.columns)


#create Database
import mysql.connector as mysql
from mysql.connector import Error

try:
    conn = mysql.connect(host='localhost', user='root', password='')
    if conn.is_connected():
        cursor = conn.cursor()
        # cursor.execute("CREATE DATABASE CovidCount")
        print("Creditcard database is created")
        
except Error as e:
    print("Error while connecting to MySQL", e)
    
    
#Import CSV into SQL   
try:
    conn = mysql.connect(host='localhost', 
                           database='Creditcard', user='root', 
                           password='')
    if conn.is_connected():
        cursor = conn.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
        cursor.execute('DROP TABLE IF EXISTS credit;')
        print('Creating table....')
        cursor.execute('''CREATE TABLE credit (Time INT(10) 
                        NOT NULL, V1 INT(5) NOT NULL,
                        V2 INT(10) NOT NULL, 
                        V3 INT(10) NOT NULL,
						V4 INT(200)NOT 
                        NULL,
						V5 INT(10)NOT NULL,
						V6 INT(10)NOT NULL,V7 INT(10) NOT NULL, 
                        V8 INT(10) NOT NULL,
						V9 INT(10)NOT 
                        NULL,V10 INT(5) NOT NULL,
                        V11 INT(10) NOT NULL, 
                        V12 INT(10) NOT NULL,
						V13 INT(200)NOT 
                        NULL,
						V14 INT(10)NOT NULL,
						V15 INT(10)NOT NULL,V16 INT(10) NOT NULL, 
                        V17 INT(10) NOT NULL,
						V18 INT(10)NOT 
                        NULL,V19 INT(5) NOT NULL,
                        V20 INT(10) NOT NULL, 
                        V21 INT(10) NOT NULL,
						V22 INT(200)NOT 
                        NULL,
						V23 INT(10)NOT NULL,
						V24 INT(10)NOT NULL,V25 INT(10) NOT NULL, 
                        V26 INT(10) NOT NULL,
						V27 INT(10)NOT 
                        NULL,V28 INT(5) NOT NULL,Amount INT(10) NOT NULL,Class INT(10) NOT NULL
						 )''')
        print("eye table is created....")
        for i,row in dataframe.iterrows():
            sql = "INSERT INTO Eyedisease.eye VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            cursor.execute(sql, tuple(row))
            # print("Record inserted")
            # the connection is not autocommitted by default, so we 
#            must commit to save our changes
            conn.commit()
except Error as e:
    print("Error while connecting to MySQL", e)
    
    
# Execute query
sql = "SELECT * FROM credit"
cursor.execute(sql)

# Fetch all the records
result = cursor.fetchall()
for i in result:
    print(i)


dataframe = pd.DataFrame(result)
dataframe.columns =['Time','V1','V2','V3','V4','V5',
'V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21'
'V22','V23','V24','V25','V26','V27','V28','Amount','Class']