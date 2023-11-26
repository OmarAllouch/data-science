import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load dataset
data = pd.read_csv("data.csv", sep=";")


# Exploratory data analysis ---------------------------------------------

# Overall summary
print(data.head())
print(data.describe())

# Search for any missing values
missing_count = data.isnull().sum()
print(missing_count)

# Define a list of categorical variables
categorical_variables = ["marital", "education", "default",
                         "housing", "loan", "contact", "poutcome", "class"]

# Plot a bar chart of all categorical variables in a single figure
plt.figure(0)
for i, column in enumerate(categorical_variables):
    plt.subplot(3, 3, i+1)
    data[column].value_counts().plot.bar()
    plt.subplots_adjust(hspace=0.75)
    plt.xticks(rotation=45)
    plt.title(column)
plt.show()

# Plot a boxplot of age and balance separately in a single figure
plt.figure(1)
plt.subplot(1, 2, 1)
data["age"].plot.box()
plt.title("age")
plt.subplot(1, 2, 2)
data["balance"].plot.box()
plt.title("balance")
plt.show()
# We can see that there are outliers in the balance variable. We will deal with them later.

# Distributions of age and balance
plt.figure(2)
sns.histplot(data['age'], kde=True)
plt.title("Histogram with Density Estimate")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

plt.figure(3)
sns.histplot(data['balance'], kde=True)
plt.title("Histogram with Density Estimate")
plt.xlabel("Balance")
plt.ylabel("Frequency")
plt.show()
# Both variables are skewed to the right.
# We will not apply any transformation in this particular case.

# Convert categorical variables to pandas categoricals
label_encoder = LabelEncoder()

for column in categorical_variables:
    data[column] = label_encoder.fit_transform(data[column])

# Correlation matrix
plt.figure(4, figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()
# We can see that there is no strong correlation between the variables.
# Except between poutcome and class, which is normal since poutcome is the outcome of the previous marketing campaign.

# -----------------------------------------------------------------------

# Data preprocessing ----------------------------------------------------

# Remove outliers
data = data[data["balance"] < 30000]

# Skewness correction
data["age"] = np.log(data["age"])
# Balance can be negative, so we will add a constant to it before applying the log transformation
data["balance"] = data["balance"] + 30000
data["balance"] = np.log(data["balance"])

# Split data into train and test
X = data.drop(columns=['class'], axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234)

# -----------------------------------------------------------------------

# Decision tree classifier ---------------------------------------------

# Baseline model
print("Baseline model: ------------------------------")
# Apply the model
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", model.score(X_test, y_test))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred))

# Plot the tree
plt.figure(1, figsize=(16, 16))
plot_tree(model, feature_names=['age', 'marital', 'education', 'default',
          'balance', 'housing', 'loan', 'contact', 'poutcome'], rounded=True, filled=True)
plt.show()

# Plot the ROC curve
plt.figure(2)
y_pred_proba = model.predict_proba(X_test)[::, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="auc=" + str(auc))
plt.legend(loc=4)
plt.show()
# Result: accuracy = 0.74, auc = 0.64
# As you can see, the model is overfitting the training data. We will try to improve it in the next section.

# Using GridSearchCV
print("GridSearchCV: ------------------------------")

model = DecisionTreeClassifier(criterion="entropy")
param_grid = {
    "max_depth": np.arange(1, 10),
    "min_samples_split": np.arange(2, 15, 2),
    "min_samples_leaf": np.arange(1, 11),
    "ccp_alpha": np.arange(0, 0.001, 0.0001)
}

grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)

# Apply the model
model = grid_search.best_estimator_
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", model.score(X_test, y_test))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred))

# Plot the tree
plt.figure(3, figsize=(16, 16))
plot_tree(model, feature_names=['age', 'marital', 'education', 'default',
          'balance', 'housing', 'loan', 'contact', 'poutcome'], rounded=True, filled=True)
plt.show()

# Plot the ROC curve
plt.figure(4)
y_pred_proba = model.predict_proba(X_test)[::, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="auc=" + str(auc))
plt.legend(loc=4)
plt.show()
# Result: accuracy = 0.82, auc = 0.80

# Post pruning
print("Post pruning: ------------------------------")

path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

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

# Accuracy vs alpha for training and testing sets
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o",
        label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o",
        label="test", drawstyle="steps-post")
ax.legend()
plt.show()

# Taking the best alpha
best_alpha = ccp_alphas[np.argmax(test_scores)]
print("Best alpha:", best_alpha)

clf = DecisionTreeClassifier(random_state=0, criterion='entropy',
                             ccp_alpha=best_alpha, max_depth=5,
                             min_samples_leaf=6, min_samples_split=2)
clf.fit(X_train, y_train)
plt.figure(5, figsize=(16, 16))
plot_tree(clf, feature_names=['age', 'marital', 'education', 'default',
          'balance', 'housing', 'loan', 'contact', 'poutcome'], rounded=True, filled=True)
plt.show()

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", clf.score(X_test, y_test))
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
# Result: accuracy = 0.82, auc = 0.80

# -----------------------------------------------------------------------

# Conclusion ------------------------------------------------------------
# The best model is the one obtained with GridSearchCV.
# It has an accuracy of 0.82 and an AUC of 0.80.
#
# As we noticed in the EDA, the poutcome variable is the most important one.
# It's the first node of the tree. It's normal since it's the outcome of the previous marketing campaign.
# If the outcome is success, the client is more likely to subscribe to the term deposit.
#
# The second most important variable is the housing variable.
# If the client has a housing loan, he is less likely to subscribe to the term deposit.
#
# The third most important variable is the balance variable.
# And so on...
#
# However on a practical level, the decision tree model as a whole is not the best.
# We will explore in Partie-IV if a Random Forest model can improve the results.
# -----------------------------------------------------------------------
