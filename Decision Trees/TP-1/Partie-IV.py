import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("data.csv", sep=";")

# Define a list of categorical variables
categorical_variables = ["marital", "education", "default",
                         "housing", "loan", "contact", "poutcome", "class"]

# Convert categorical variables to pandas categoricals
label_encoder = LabelEncoder()

for column in categorical_variables:
    data[column] = label_encoder.fit_transform(data[column])


# Split data into train and test
X = data.drop(columns=['class'], axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234)


# XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
