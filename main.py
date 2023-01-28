# This is an ML model that will predict whether a person is diabetic or not.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Creating Dataframe
df = pd.read_csv('diabetes.csv')
# print(df)

# finding no. of rows and columns
# print(df.shape)  (768 rows , 9 columns)

# Statistical Analysis
# print(df.describe())

# Count of no. of diabetic and non-diabetic patients
# print(df.Outcome.value_counts())
# 500 - non diabetic peoples
# 268 - diabetic peoples

# This line of code was written to group the data on basis of outcome and mean values of parameters for respective outcomes
# print(df.groupby('Outcome').mean())

# Preparation of data for training the model
x = df.drop('Outcome', axis=1)
y = df.Outcome

# Standardisation of data
scaler = StandardScaler()
scaler.fit(x)
standardised_data = scaler.transform(x)

x = standardised_data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# Training the model using support vector machine algorithm
model = SVC()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# Accuracy of model
print("Accuracy of the model is : ", accuracy)

