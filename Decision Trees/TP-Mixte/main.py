import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.ensemble import IsolationForest

# Load dataset
data = pd.read_csv('Income_Inequality.csv', sep=';')

# Remove the identity columns
data = data.drop(['Country', 'Year'], axis=1)

# # Exploratory Data Analysis --------------------------------------------
# # Print the first 5 rows of the dataframe.
# print("Head of the dataframe:")
# print(data.head())

# # Print the shape of the dataframe
# print("Shape of the dataframe:")
# print(data.shape)

# # Print the missing values of the dataframe
# print("Missing values of the dataframe:")
# print(data.isna().sum())
# # There are no missing values in the dataframe

# # Print the summary statistics of the dataframe
# print("Summary statistics of the dataframe:")
# print(data.describe())

# # Print the unique values of the categorical variables
categorical_variables = ['Income_Inequality']
# print("Unique values of the categorical variables:")
# print(data[categorical_variables].nunique())

# # Plot boxplots for numerical variables in a single figure
# economical_variables = ['Eco1', 'Eco2', 'Eco3']

# plt.figure(0)
# for i, column in enumerate(economical_variables):
#     plt.subplot(1, 3, i+1)
#     data[column].plot.box()
#     plt.subplots_adjust(hspace=0.75)
#     plt.xticks(rotation=0)
#     plt.title(column)
# plt.show()

# energy_variables = ['Energy1', 'Energy2', 'Energy3']

# plt.figure(1)
# for i, column in enumerate(energy_variables):
#     plt.subplot(1, 3, i+1)
#     data[column].plot.box()
#     plt.subplots_adjust(hspace=0.75)
#     plt.xticks(rotation=0)
#     plt.title(column)
# plt.show()

# health_variables = ['Health1', 'Health2']

# plt.figure(2)
# for i, column in enumerate(health_variables):
#     plt.subplot(1, 2, i+1)
#     data[column].plot.box()
#     plt.subplots_adjust(hspace=0.75)
#     plt.xticks(rotation=0)
#     plt.title(column)
# plt.show()


# financial_variables = ['Finan1', 'Finan2', 'Finan3', 'Finan4', 'Finan5']

# plt.figure(3)
# for i, column in enumerate(financial_variables):
#     plt.subplot(1, 5, i+1)
#     data[column].plot.box()
#     plt.subplots_adjust(hspace=0.75)
#     plt.xticks(rotation=0)
#     plt.title(column)
# plt.show()

# gov_pov_env = ['Governance', 'Poverty', 'Env']

# plt.figure(4)
# for i, column in enumerate(gov_pov_env):
#     plt.subplot(1, 3, i+1)
#     data[column].plot.box()
#     plt.subplots_adjust(hspace=0.75)
#     plt.xticks(rotation=0)
#     plt.title(column)
# plt.show()

# other_variables = ['Other1', 'Other2', 'Other3']

# plt.figure(5)
# for i, column in enumerate(other_variables):
#     plt.subplot(1, 3, i+1)
#     data[column].plot.box()
#     plt.subplots_adjust(hspace=0.75)
#     plt.xticks(rotation=0)
#     plt.title(column)
# plt.show()

# # ----------------------------------------------------------------------

# Data preprocessing ---------------------------------------------------

# Encode categorical variables
label_encoder = LabelEncoder()

for column in categorical_variables:
    data[column] = label_encoder.fit_transform(data[column])

# # Correlation matrix
# plt.figure(6, figsize=(10, 10))
# sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
# plt.show()
# Remove correlated variables

# Split the data into train and test sets
X = data.drop('Income_Inequality', axis=1)
y = data['Income_Inequality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# ----------------------------------------------------------------------

# Model training -------------------------------------------------------

# Decision tree classifier
# Create a decision tree classifier
dt = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Train the classifier
dt.fit(X_train, y_train)

# Evaluate the classifier
print("\nAccuracy of the classifier on test set:", dt.score(X_test, y_test))

# Plot the tree
plt.figure(7, figsize=(15, 15))
plot_tree(dt, filled=True, rounded=True, class_names=[
          'Low', 'High'], feature_names=X.columns.to_list())
plt.show()

# Grid search
param_grid = {
    'max_depth': np.arange(2, 12),
    'min_samples_leaf': np.arange(2, 10),
    'min_samples_split': np.arange(2, 10),
    'ccp_alpha': np.arange(0, 0.5, 0.01)
}

grid_search = GridSearchCV(DecisionTreeClassifier(criterion='entropy',
                                                  random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

model = grid_search.best_estimator_
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nAccuracy:", model.score(X_test, y_test))
print("Tree size:", model.tree_.node_count)
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred))

# Plot the tree
plt.figure(8, figsize=(15, 15))
plot_tree(model, filled=True, rounded=True, class_names=[
          'Low', 'High'], feature_names=X.columns.to_list())
plt.show()

# ROC curve
plt.figure(2)
y_pred_proba = model.predict_proba(X_test)[::, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="auc=" + str(auc))
plt.legend(loc=4)
plt.show()
# Result: accuracy = 0.9578, auc = 0.9652

# Post pruning
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]


# Taking the best alpha
test_scores = [clf.score(X_test, y_test) for clf in clfs]
best_alpha = ccp_alphas[np.argmax(test_scores)]
print("Best alpha:", best_alpha)

clf = DecisionTreeClassifier(random_state=0, criterion='entropy',
                             ccp_alpha=best_alpha, max_depth=5,
                             min_samples_leaf=6, min_samples_split=2)
clf.fit(X_train, y_train)
plt.figure(5, figsize=(16, 16))
plot_tree(clf, feature_names=X.columns.to_list(), rounded=True, filled=True)
plt.show()

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", clf.score(X_test, y_test))
print("Tree size:", clf.tree_.node_count)
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred))

# Plot the ROC curve
plt.figure(6)
y_pred_proba = clf.predict_proba(X_test)[::, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="auc=" + str(auc))
plt.legend(loc=4)
plt.show()
# Result: accuracy = 0.9425, auc = 0.9606

# Variable importance
plt.figure(9, figsize=(10, 10))
plt.barh(X.columns.to_list(), clf.feature_importances_)
plt.title("Variable importance")
plt.show()

# Use the most important variables
X_important = data[['Eco1', 'Eco2', 'Health2', 'Finan3', 'Finan4',
                    'Finan5', 'Governance', 'Poverty', 'Other2', 'Other3']]

X_important_train, X_important_test, y_train, y_test = train_test_split(
    X_important, y, test_size=0.3, random_state=42)

clf.fit(X_important_train, y_train)
print("Accuracy:", clf.score(X_important_test, y_test))
# Result: accuracy = 0.9463

# ----------------------------------------------------------------------

# Conclusion -----------------------------------------------------------

# The model did very well, with an accuracy of 0.9578 and an AUC of 0.9652
# before pruning, and an accuracy of 0.9425 and an AUC of 0.9606 after pruning.

# After finding the most important variables, and applying the model to
# only those variables, the accuracy was 0.9463, which is very close to the
# accuracy of the model with all the variables.

# The most important variables are: Eco1, Eco2, Health2, Finan3, Finan4,
# Finan5, Governance, Poverty, Other2, Other3.

# ----------------------------------------------------------------------


# Isolation Forest -----------------------------------------------------

# Use entire dataset
X = data.drop('Income_Inequality', axis=1)
isolation_forest = IsolationForest(random_state=1234, n_estimators=100)
isolation_forest.fit(X)

# Predict anomalies
y_pred = isolation_forest.predict(X)

# Calculate anomaly score and depth
scores = isolation_forest.decision_function(X)
scores = pd.DataFrame(scores)
scores = scores.sort_values(by=0, ascending=False)

# Print top 10 and bottom 10
print("Top 10 anomalies:")
print(scores.head(10))
print("\nBottom 10 anomalies:")
print(scores.tail(10))

# Get original points
anomalies = X.iloc[scores.head(10).index]
print("\nOriginal points of top 10 anomalies:")
print(anomalies)

# Compare the difference between each anomaly and the mean to the standard deviation
print("\nDifference between each anomaly and the standard deviation:")
print((anomalies - X.mean()) / X.std())

# Print the max across all variables for each anomaly, and get the name of the variable
print("\nMax across all variables for each anomaly:")
print(anomalies.max(axis=1))
print("\nName of the variable:")
print(anomalies.idxmax(axis=1))

# Detect first 50 anomalies
print("\nFirst 50 anomalies:")
print(scores.head(50))

# Remove first 50 anomalies from the dataset
X_clean = X.drop(scores.head(50).index)
y_clean = y.drop(scores.head(50).index)

# Apply the previous classifier to the clean dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=42)

clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))
y_pred = clf.predict(X_test)
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred))
# Result: accuracy = 0.9471

# ----------------------------------------------------------------------
