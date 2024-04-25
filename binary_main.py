from IPython.display import display, HTML, Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import make_scorer
from sklearn import metrics
from Binary_Relevance_Algorithm import BinaryRelevanceClassifier
from accuracy import accuracy_score
# to avoid future warnings for sklearn
import warnings
warnings.filterwarnings("ignore")

#加载数据集

# Read the CSV file
dataset = pd.read_csv('data/yeast.csv')
print("Dataset.shape: " + str(dataset.shape))

# split the features-X and class labels-y
X = dataset.iloc[:, :103]
y = dataset.iloc[:, 103:]

print("X.shape: " + str(X.shape))
display(X.head())
print("y.shape: " + str(y.shape))
display(y.head())
print("Descriptive stats:")
X.describe()

# Normalise the data
X = (X-X.min())/(X.max()-X.min())

# 划分测试集训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)

print("X_train.shape: " + str(X_train.shape))
print("X_test.shape: " + str(X_test.shape))
print("y_train.shape: " + str(y_train.shape))
print("y_test.shape: " + str(y_test.shape))


# # instantiate the classifier
# br_clf = BinaryRelevanceClassifier(RandomForestClassifier(criterion='entropy'))
# # fit
# br_clf.fit(X_train, y_train)
# # predict
# y_pred = br_clf.predict(X_test)
# print("y_pred.shape: " + str(y_pred.shape))
#
# print("Accuracy of Binary Relevance Classifier: " + str(accuracy_score(y_test, y_pred)))
# cv_folds=5
#
# # Set up the parameter grid to search
# param_grid ={'base_classifier': [DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=2),
#                                  RandomForestClassifier(criterion='entropy'),
#                                  LogisticRegression(max_iter=20000), GaussianNB(), KNeighborsClassifier(), SVC()] }
#
# # Perform the search
# # Using the custom accuracy function defined earlier
# tuned_model = GridSearchCV(BinaryRelevanceClassifier(), \
#                             param_grid, scoring=make_scorer(accuracy_score), verbose = 2, n_jobs = -1, cv=cv_folds)
# tuned_model.fit(X_train, y_train)
#
# # Print details of the best model
# print("Best Parameters Found: ")
# display(tuned_model.best_params_)
# display(tuned_model.best_score_)

# list of base models
base_models = [DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=2),
               RandomForestClassifier(criterion='entropy'),
               LogisticRegression(max_iter=20000), GaussianNB(), KNeighborsClassifier(), SVC()]
base_model_names = ["Decision Tree", "Random Forest", "Logistic Regression", "GaussianNB", "kNN", "SVM"]

# store accuracy scores
br_clf_accuracies = dict()
br_clfus_accuracies = dict()

# store F1 scores
br_clf_f1 = dict()
br_clfus_f1 = dict()

i = 0
for clf in base_models:
    # without undersampling
    br_clf = BinaryRelevanceClassifier(clf)
    br_clf.fit(X_train, y_train)
    br_y_pred = br_clf.predict(X_test)

    # find accuracy using custom accuracy function defined
    accuracy = accuracy_score(y_test, br_y_pred)
    br_clf_accuracies[base_model_names[i]] = accuracy

    # find f1 score using sklearn
    y_pred_df = pd.DataFrame(br_y_pred)
    f1_score_br = metrics.f1_score(y_test, y_pred_df, average='macro')
    br_clf_f1[base_model_names[i]] = f1_score_br



    i += 1

print("===================Accuracy Scores=====================")
print("Binary Relevance")
display(br_clf_accuracies)

print("======================F1 Scores========================")
print("Binary Relevance")
display(br_clf_f1)

# 绘制accuracy图
plt.plot(list(br_clf_accuracies.keys()), list(br_clf_accuracies.values()))
plt.xticks(rotation=90)
plt.legend(['Binary Relevance'])
plt.tight_layout()
plt.savefig('binary_relevance_accuracies.png')
plt.show()
# 绘制F1图
plt.plot(list(br_clf_f1.keys()), list(br_clf_f1.values()))
plt.xticks(rotation=90)
plt.legend(['Binary Relevance'])
plt.tight_layout()
plt.savefig('binary_relevance_f1.png')
plt.show()