import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# Load your dataset
data = pd.read_csv("data.csv", sep=";")

# Define a list of categorical variables
categorical_variables = ["marital", "education", "default",
                         "housing", "loan", "contact", "poutcome", "class"]

# Convert categorical variables to pandas categoricals
label_encoder = LabelEncoder()

for column in categorical_variables:
    data[column] = label_encoder.fit_transform(data[column])


# # Exploratory data analysis
# # Overall summary
# print(data.describe())

# # Search for any missing values
# missing_count = data.isnull().sum()
# print(missing_count)  # No missing data in our database

# # Distributions of observations

# sns.histplot(data['balance'], kde=True)
# plt.title("Histogram with Density Estimate")
# plt.xlabel("Balance")
# plt.ylabel("Frequency")
# plt.show()

# Split data into train and test
np.random.seed(1234)
X = data.drop(columns=['class'])
y = data['class']
X_train, X_test = train_test_split(X, test_size=0.3)
y_train, y_test = train_test_split(y, test_size=0.3)

# # Define the hyperparameter grid
# param_grid = {
#     "min_samples_split": np.arange(2, 15, 2),
#     "min_samples_leaf": np.arange(1, 11),
#     # lower and upper bound of `alpha` were found by a bit of manual trial and error
#     "ccp_alpha": np.arange(0, 0.0031, 0.0005)
# }

# model = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
# model.fit(X_train, y_train)

# best_model = model.best_estimator_
# best_params = model.best_params_

# y_pred = best_model.predict(X_test)
# print("Best Hyperparameters:", best_params)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# Custom model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
