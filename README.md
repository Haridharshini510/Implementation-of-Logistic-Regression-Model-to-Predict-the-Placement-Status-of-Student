# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Haridharshini J
RegisterNumber: 212224040098 
*/

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("Placement_Data.csv")
print(data.head())

# Copy and drop unnecessary columns
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)
print(data1.head())

# Check nulls and duplicates
print(data1.isnull().sum())
print("Duplicates:", data1.duplicated().sum())

# Label encoding
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# Features and target
x = data1.iloc[:, :-1]
y = data1["status"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# Logistic Regression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

# Predictions
y_pred = lr.predict(x_test)
print("Predictions:", y_pred)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)

# Predict for a sample input
sample_pred = lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
print("Sample Prediction:", sample_pred)

```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
<img width="791" height="155" alt="image" src="https://github.com/user-attachments/assets/88ccb859-4657-4dcb-8335-5c4ce30f7f14" />
<img width="787" height="163" alt="image" src="https://github.com/user-attachments/assets/0aab5361-8a59-4394-9722-e76f8ed11fd7" />
<img width="813" height="162" alt="image" src="https://github.com/user-attachments/assets/c63d533d-77f1-4fc6-8b8d-0895620a8a85" />
<img width="597" height="169" alt="image" src="https://github.com/user-attachments/assets/7e27c3f7-20da-4354-bdeb-16b7699c81f3" />
<img width="955" height="50" alt="image" src="https://github.com/user-attachments/assets/7e6c25f7-e1ff-4b1a-b27b-52d03fa62d93" />
<img width="668" height="231" alt="image" src="https://github.com/user-attachments/assets/97a1499c-1bd7-4904-9c43-4bb91ba15d4c" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
