"""Import the libraries"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix
#import scikitplot as skplt
import warnings
warnings.filterwarnings("ignore")

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
        print("credit table is created....")
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
#===============================INPUT==========================================
"""Input Data Read"""
data = pd.read_csv('creditcard_1.csv')
print(data.head())
print(data.info())
print(data.describe())

#==============================PRE-PROCESSING==================================
"""Preprocessing"""

"""Checking missing values in the data"""
print()
print("Checking Missing Values")
print(data.isnull().sum())
data.describe()

#==========================EXPLARATORY DATA ANALYSIS===========================
"""Data visualization"""
"""number of fraud and valid transactions """

count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency");



"""Assigning the transaction class "0 = NORMAL  & 1 = FRAUD"""

Normal = data[data['Class']==0]
Fraud = data[data['Class']==1]
print()
print("Outlier Fraction:", len(Fraud)/float(len(Normal)))
print()
print("Fraud Cases : {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Normal)))


#==============================MODEL SELECTION=================================

X = data.iloc[:,:-1] 
y = data.iloc[:,-1]

"""Splitting train and test data"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#=================================CLASSIFICATION===============================
"""Naive Bayes Classifier"""

nb = GaussianNB()
nb.fit(X_train,y_train)

nb_pred=nb.predict(X_test)
print('\n')
print("------Accuracy------")
nb=accuracy_score(y_test, nb_pred)*100
NB=('Naive Bayes Accuracy is:',nb,'%')
print(NB)
print('\n')
print("------Classification Report------")
print(classification_report(nb_pred,y_test))
print('\n')
print('Confusion_matrix')
nb_cm = confusion_matrix(y_test, nb_pred)
print(nb_cm)
print('\n')
tn = nb_cm[0][0]
fp = nb_cm[0][1]
fn = nb_cm[1][0]
tp = nb_cm[1][1]
Total_TP_FP=nb_cm[0][0]+nb_cm[0][1]
Total_FN_TN=nb_cm[1][0]+nb_cm[1][1]
specificity = tn / (tn+fp)
nb_specificity=format(specificity,'.3f')
print('NB_specificity:',nb_specificity)
print()

#plt.figure()
#skplt.estimators.plot_learning_curve(GaussianNB(), X_train, y_train,
#                                     cv=7, shuffle=True, scoring="accuracy",
#                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
#                                     title="Naive Bayes Digits Classification Learning Curve");

plt.figure()                                   
sns.heatmap(confusion_matrix(y_test,nb_pred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


'''RANDOM FOREST'''

rf_clf=RandomForestClassifier(n_estimators=3)
rf_clf.fit(X_train,y_train)

rf_ypred=rf_clf.predict(X_test)
print('\n')
print("------Accuracy------")
rf=accuracy_score(y_test, rf_ypred)*100
RF=('RANDOM FOREST Accuracy:',accuracy_score(y_test, rf_ypred)*100,'%')
print(RF)
print('\n')
print("------Classification Report------")
print(classification_report(rf_ypred,y_test))
print('\n')
print('Confusion_matrix')
rf_cm = confusion_matrix(y_test, rf_ypred)
print(rf_cm)
print('\n')
tn = rf_cm[0][0]
fp = rf_cm[0][1]
fn = rf_cm[1][0]
tp = rf_cm[1][1]
Total_TP_FP=rf_cm[0][0]+rf_cm[0][1]
Total_FN_TN=rf_cm[1][0]+rf_cm[1][1]
specificity = tn / (tn+fp)
rf_specificity=format(specificity,'.3f')
sensitivity = tp / (fn + tp)
rf_sensitivity=format(sensitivity,'.3f')
print('RF_specificity:',rf_specificity)
print('\n')

#plt.figure()
#skplt.estimators.plot_learning_curve(RandomForestClassifier(), X_train, y_train,
#                                     cv=7, shuffle=True, scoring="accuracy",
#                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
#                                     title="Random Forest Digits Classification Learning Curve");

plt.figure()                                   
sns.heatmap(confusion_matrix(y_test,rf_ypred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


#comparision
vals=[nb,rf]
inds=range(len(vals))
labels=["NB ","RF"]
fig,ax = plt.subplots()
rects = ax.bar(inds, vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.show()
