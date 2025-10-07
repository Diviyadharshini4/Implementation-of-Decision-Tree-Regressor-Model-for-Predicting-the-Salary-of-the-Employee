# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset. 

4.Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: A.DIVIYADHARSHINI
RegisterNumber: 212224040080 
*/
```
```

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:

Data Head:

<img width="483" height="313" alt="image" src="https://github.com/user-attachments/assets/7371aba7-1125-4faa-ac51-69f595a136e3" />

Data Info:

<img width="773" height="301" alt="image" src="https://github.com/user-attachments/assets/517c31aa-a3f4-4f37-b5f2-127ceaf1b001" />

isnull() sum():


<img width="194" height="105" alt="image" src="https://github.com/user-attachments/assets/a0a25daa-7b5f-4b87-9e80-b08852d1385b" />

Data Head for salary:

<img width="320" height="280" alt="image" src="https://github.com/user-attachments/assets/71436269-6d16-495b-9a1d-384387d43092" />

Mean Squared Error :

<img width="146" height="38" alt="image" src="https://github.com/user-attachments/assets/11ccf6f3-1bed-4ed2-8d17-da0026459eb2" />

r2 Value:

<img width="188" height="34" alt="image" src="https://github.com/user-attachments/assets/8b01c84b-7b38-4805-86cb-30305cc7a7c1" />

Data prediction :

<img width="194" height="39" alt="image" src="https://github.com/user-attachments/assets/650c01d6-3b71-4f05-a11a-2d999ee885d5" />











## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
