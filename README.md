# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Gokkul M
RegisterNumber: 212223240039

import chardet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
print(result)
data=pd.read_csv("spam.csv",encoding='Windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())
x=data["v1"].values
y=data["v2"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
svc=SVC()
print(svc.fit(x_train,y_train))
y_pred=svc.predict(x_test)
print(y_pred)
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
```

## Output:
![image](https://github.com/Gokkul-M/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870543/1fcb6e3a-348d-42a5-86a3-e26db43f0bae)
![image](https://github.com/Gokkul-M/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870543/fe431058-12ea-4604-ae1e-22ce949ed74e)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
