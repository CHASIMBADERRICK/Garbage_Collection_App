# Classification model to predict whether a person makes over $50k a year

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# from google.colab import drive
# drive.mount('/drive')

# Collect the data

income_data = pd.read_csv("/home/derrick/Documents/adult.csv")
income_data.head()

# Pre-process the data

# Naming the Columns

income_data.columns = ['age', 'workclass', 'fnlwgt', 
           'education', 'education-num', 
           'marital-status', 'occupation', 
           'relationship', 'race', 'sex', 
           'capital-gain', 'capital-loss', 
           'hours-per-week', 'native-country', 
           'income']


income_data.head()

# Removing null values from the dataset
income_data.isnull().sum()

# Split the Data

X = income_data.drop('income', axis=1)
Y = income_data['income']

# encode categorical variables using one-hot encoding
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_features)], remainder='passthrough')
X = preprocessor.fit_transform(X)

# split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# a.Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

# make predictions on the test data
dt_pred = model.predict(X_test)

print("Confusion matrix for Decision Tree:")
cm = confusion_matrix(Y_test, dt_pred)
print(confusion_matrix(Y_test, dt_pred))

print("\nClassification report for Decision Tree:")
print(classification_report(Y_test, dt_pred))

precision = cm[1,1] / (cm[1,1] + cm[0,1])
print("Precision: {:.2f}".format(precision))

recall = cm[1,1] / (cm[1,1] + cm[1,0])
print("Recall   : {:.2f}".format(recall))

f1_score = 2 * precision * recall / (precision + recall)
print("F1-score : {:.2f}".format(f1_score))

dt_accuracy = accuracy_score(Y_test, dt_pred)
print("Accuracy : {:.2f}".format(dt_accuracy))

# Percentage of Misclassification

dt_misclassification = 100 - (dt_accuracy * 100)
print("Misclassification: {:.2f}%".format(dt_misclassification))

# b.Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)

print("\nConfusion matrix for Random Forest:")
cm = confusion_matrix(Y_test, rf_pred)
print(confusion_matrix(Y_test, rf_pred))

print("\nClassification report for Random Forest:")
print(classification_report(Y_test, rf_pred))

precision = cm[1,1] / (cm[1,1] + cm[0,1])
print("Precision: {:.2f}".format(precision))

recall = cm[1,1] / (cm[1,1] + cm[1,0])
print("Recall   : {:.2f}".format(recall))

f1_score = 2 * precision * recall / (precision + recall)
print("F1-score : {:.2f}".format(f1_score))

rf_accuracy = accuracy_score(Y_test, rf_pred)
print("Accuracy : {:.2f}".format(rf_accuracy))

# Percentage of Misclassification

rf_misclassification = 100 - (rf_accuracy * 100)
print("Misclassification: {:.2f}%".format(rf_misclassification))

# c.Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
lr_pred = lr.predict(X_test)

print("\nConfusion matrix for Logistic Regression:")
cm = confusion_matrix(Y_test, lr_pred)
print(confusion_matrix(Y_test, lr_pred))

print("\nClassification report for Logistic Regression:")
print(classification_report(Y_test, lr_pred))

precision = cm[1,1] / (cm[1,1] + cm[0,1])
print("Precision: {:.2f}".format(precision))

recall = cm[1,1] / (cm[1,1] + cm[1,0])
print("Recall   : {:.2f}".format(recall))

f1_score = 2 * precision * recall / (precision + recall)
print("F1-score : {:.2f}".format(f1_score))

lr_accuracy = accuracy_score(Y_test, lr_pred)
print("Accuracy : {:.2f}".format(lr_accuracy))

# Percentage of Misclassification

lr_misclassification = 100 - (lr_accuracy * 100)
print("Misclassification: {:.2f}%".format(lr_misclassification))

# d.KNN Classifier

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knn_pred = knn.predict(X_test)

print("\nConfusion matrix for KNN:")
cm = confusion_matrix(Y_test, knn_pred)
print(confusion_matrix(Y_test, knn_pred))

print("\nClassification report for KNN:")
print(classification_report(Y_test, knn_pred))

precision = cm[1,1] / (cm[1,1] + cm[0,1])
print("Precision: {:.2f}".format(precision))

recall = cm[1,1] / (cm[1,1] + cm[1,0])
print("Recall   : {:.2f}".format(recall))

f1_score = 2 * precision * recall / (precision + recall)
print("F1-score : {:.2f}".format(f1_score))

knn_accuracy = accuracy_score(Y_test, knn_pred)
print("Accuracy : {:.2f}".format(knn_accuracy))

# Percentage of Misclassification

knn_misclassification = 100 - (knn_accuracy * 100)
print("Misclassification: {:.2f}%".format(knn_misclassification))

# e.SVM Classifier

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, Y_train)
svm_pred = svm.predict(X_test)

print("\nConfusion matrix for SVM:")
cm = confusion_matrix(Y_test, svm_pred)
print(confusion_matrix(Y_test, svm_pred))

print("\nClassification report for SVM:")
print(classification_report(Y_test, svm_pred))

precision = cm[1,1] / (cm[1,1] + cm[0,1])
print("Precision: {:.2f}".format(precision))

recall = cm[1,1] / (cm[1,1] + cm[1,0])
print("Recall   : {:.2f}".format(recall))

f1_score = 2 * precision * recall / (precision + recall)
print("F1-score : {:.2f}".format(f1_score))

svm_accuracy = accuracy_score(Y_test, svm_pred)
print("Accuracy : {:.2f}".format(svm_accuracy))

# Percentage of Misclassification

svm_misclassification = 100 - (svm_accuracy * 100)
print("Misclassification: {:.2f}%".format(svm_misclassification))

# Accuracy of the Models

data =  {
    "Decision Tree"         : dt_accuracy,
    "Random Forest"         : rf_accuracy,
    "Logistic Regression"   : lr_accuracy,
    "KNN"                   : knn_accuracy,
    "SVM"                   : svm_accuracy,
    
}

courses = list(data.keys())
values  = list(data.values())
fig = plt.figure(figsize = (10, 5))

plt.bar(courses, values,color ='brown',width = 0.4)
plt.xlabel("Models")
plt.ylabel("Testing accuracy")
plt.title("Testing accuracy v/s Models")
plt.show()

# Determine the model with the best accuracy
best_model = max(dt_accuracy, rf_accuracy, knn_accuracy, lr_accuracy, svm_accuracy)

if best_model == dt_accuracy:
    print("\nThe best model is Decision Tree.")
elif best_model == rf_accuracy:
    print("\nThe best model is Random Forest.")
elif best_model == knn_accuracy:
    print("\nThe best model is KNN.")
elif best_model == lr_accuracy:
    print("\nThe best model is Logistic Regression.")
# elif best_model == svm_accuracy:
#     print("\nThe best model is SVM.