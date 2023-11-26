import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234)

# selector = SelectKBest(k=5)
# selector.fit_transform(X_train, y_train)
# feature_names = selector.get_feature_names_out()
# print("Best 5 features:", feature_names)

# X_train = X_train[feature_names]
# X_test = X_test[feature_names]

# Define the hyperparameter grid
param_grid = {
    "criterion": ["entropy"],
    "min_samples_split": np.arange(2, 15, 2),
    "min_samples_leaf": np.arange(1, 11),
    # lower and upper bound of `alpha` were found by a bit of manual trial and error
    "ccp_alpha": np.arange(0, 0.001, 0.0001)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=kf)
model.fit(X_train, y_train)

best_model = model.best_estimator_
best_params = model.best_params_

y_pred = best_model.predict(X_test)
print("Best Hyperparameters:", best_params)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(16, 16))
plot_tree(best_model, feature_names=['age', 'marital', 'education', 'default',
          'balance', 'housing', 'loan', 'contact', 'poutcome'], rounded=True, filled=True)
plt.show()

path = best_model.cost_complexity_pruning_path(X_train, y_train)
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

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

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
clf = DecisionTreeClassifier(random_state=0, criterion='entropy',
                             ccp_alpha=0.00036, min_samples_split=2,
                             min_samples_leaf=9)
clf.fit(X_train, y_train)
plt.figure(figsize=(16, 9))
plot_tree(clf, feature_names=['age', 'marital', 'education', 'default',
          'balance', 'housing', 'loan', 'contact', 'poutcome'], rounded=True, filled=True)
plt.show()

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

y_pred_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc=4)
plt.show()
