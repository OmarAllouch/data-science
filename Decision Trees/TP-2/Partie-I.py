# Import libraries
import numpy as np
import pandas as pd
import statistics

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Importing the datasets
train = pd.read_csv('exemple_code.csv', sep=';')
X_test = pd.read_csv('mot_code.csv', sep=';')


# Data preprocessing
# Remove the last column of X_test since it is empty
X_test = X_test.iloc[:, :-1]

# Separate the data into X and y
X_train = train.drop(columns=['y'], axis=1)
y_train = train['y']

# Encoding categorical data
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)

# Fit the model
dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(X_train, y_train)

# Predict the test set results
y_pred = dt.predict(X_test)

# Inverse transform the predictions
y_pred = labelencoder.inverse_transform(y_pred)

# Reshape the result to (10, 16) since each word is encoded into 16 numbers, 10 times.
y_pred = y_pred.reshape(10, 16)

# Calculate the mode for every column
modes = [statistics.mode(column) for column in y_pred.transpose()]

# Find the secret word
code_map = {"04": "a", "08": "b", "12": "c", "16": "d", "20": "e",
            "24": "f", "28": "g", "32": "h", "36": "i", "40": "j",
            "44": "k", "48": "l", "52": "m", "56": "n", "60": "o",
            "64": "p", "68": "q", "72": "r", "76": "s", "80": "t",
            "84": "u", "88": "v", "92": "w", "96": "x"}

secret_word = ""
for i in range(0, len(modes), 2):
    secret_word += code_map[str(modes[i]) + str(modes[i + 1])]
print(secret_word)

# Knowing the answer is "toulouse", we will build y_test in order to evaluate the model
y_test = np.array(list("toulouse"))
y_test = [list(code_map.keys())[list(code_map.values()).index(c)]
          for c in y_test]
y_test = np.array([[int(c[0]), int(c[1])] for c in y_test])
y_test = y_test.flatten()
y_test = np.array([y_test for _ in range(10)])

# Evaluate the model
print("Confusion matrix:")
print(confusion_matrix(y_test.flatten(), y_pred.flatten()))
print("Accuracy score:")
print(accuracy_score(y_test.flatten(), y_pred.flatten()))
print("Classification report:")
print(classification_report(y_test.flatten(), y_pred.flatten(), zero_division=0))
# The accuracy is 70.625%

# We will now try to improve the model by using GridSearchCV to find the best parameters

# Define the parameters
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4, 6, 8, 10, 12, 14, 16],
    'min_samples_split': np.arange(2, 11, 2),
    'min_samples_leaf': np.arange(6),
    'ccp_alpha': np.arange(0, 0.1, 0.01)
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=dt,
                           param_grid=param_grid,
                           cv=10,
                           n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
print("Best parameters:")
print(grid_search.best_params_)

# Predict the test set results
y_pred = grid_search.predict(X_test)

# Inverse transform the predictions
y_pred = labelencoder.inverse_transform(y_pred)

# Reshape the result to (10, 16)
y_pred = y_pred.reshape(10, 16)

# Calculate the mode for every column
modes = [statistics.mode(column) for column in y_pred.transpose()]

# Find the secret word
secret_word = ""
for i in range(0, len(modes), 2):
    secret_word += code_map[str(modes[i]) + str(modes[i + 1])]
print(secret_word)
# We got the correct answer

# Use previously generated y_test to evaluate the model
print("Confusion matrix:")
print(confusion_matrix(y_test.flatten(), y_pred.flatten()))
print("Accuracy score:")
print(accuracy_score(y_test.flatten(), y_pred.flatten()))
print("Classification report:")
print(classification_report(y_test.flatten(), y_pred.flatten(), zero_division=0))
# The accuracy is 74.375%, not a big improvement, but the precision of each class is better
