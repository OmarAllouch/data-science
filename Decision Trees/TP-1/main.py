import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

# 1
# Import the CSV file
data = pd.read_csv("data.csv", sep=";", header=0)

# 2
# COnvert categorical variables to pandas categoricals
categorical_variables = ["marital", "education", "default",
                         "housing", "loan", "contact", "poutcome", "class"]
label_encoder = LabelEncoder()

for column in categorical_variables:
    data[column] = label_encoder.fit_transform(data[column])


# Test and train sets
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234)
print(X_train.shape)
print(X_test.shape)

# Arbre maximal et élagage
Tmax = DecisionTreeClassifier(
    criterion='entropy', max_depth=30, random_state=1234)
Tmax.fit(X_train, y_train)
y_pred = Tmax.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Arbre optimisé par validation croisé
param_grid = {'max_depth': np.arange(3, 30)}
T_opt = GridSearchCV(DecisionTreeClassifier(
    criterion='entropy', random_state=1234), param_grid, cv=5)
T_opt.fit(X_train, y_train)
print(T_opt.best_params_)
y_pred = T_opt.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# 3
y_pred = T_opt.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

pred_prob = T_opt.predict_proba(X_test)
pred_prob = pd.DataFrame(pred_prob)
pred_prob = pred_prob[1]
fpr, tpr, thresholds = roc_curve(y_test, pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Area under curve
auc = roc_auc_score(y_test, pred_prob)
print('AUC: %.3f' % auc)

# Precision recall curve
precision, recall, thresholds = precision_recall_curve(y_test, pred_prob)
auc = auc(recall, precision)
print('AUC: %.3f' % auc)
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
plt.plot(recall, precision, marker='.')
plt.show()

# Average precision score
ap = average_precision_score(y_test, pred_prob)
print('AP: %.3f' % ap)
