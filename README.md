# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2.Analyse the data.
3.Use modelselection and Countvectorizer to preditct the values.
4.Find the accuracy and display the result. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: ALMAAS JAHAAN M
RegisterNumber:  212224230016
*/

import chardet
file = "C:\\Users\\admin\\Downloads\\spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

import pandas as pd
data = pd.read_csv( "C:\\Users\\admin\\Downloads\\spam.csv", encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())

x = data["v2"].values  # messages
y = data["v1"].values  # labels

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Output:
![SVM For Spam Mail Detection](sam.png)
![image](https://github.com/user-attachments/assets/05e527a6-6731-4f1e-a130-e558407a4ebd)
![image](https://github.com/user-attachments/assets/86c8e92d-a9c3-4bbd-b5bc-d085c153b899)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
