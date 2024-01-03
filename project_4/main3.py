import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets

def missingValueGen(df, missingPercent):
  totalValues = df.size
  missingValues = int(totalValues * missingPercent)
  np.random.seed(777)
  rows = np.random.randint(0, df.shape[0], size=missingValues)
  columns = np.random.randint(0, df.shape[1], size=missingValues)

  for row, col in zip(rows, columns):
    df.iat[row,col] = np.nan

  return df

XdataMissingList ={}
for missingPercent in np.arange(0, 0.8001, step=0.1):
  missingPercent = np.round(missingPercent, 2)
  XdataMissing = missingValueGen(X.copy(), missingPercent)
  XdataMissingList[missingPercent] = XdataMissing


xTrain, xTest, yTrain, yTest = train_test_split(XdataMissingList[0.8], y, train_size=0.7, random_state=777)

xTrainObj = {}
xTestObj = {}
yTrainObj = {}
yTestObj = {}

for percentValue in np.arange(0, 0.8001, step=0.1):
  percentValue = np.round(percentValue, 2)
  xTrain, xTest, yTrain, yTest = train_test_split(XdataMissingList[percentValue], y, train_size=0.7, random_state=777)
  xTrainObj[percentValue] = xTrain
  xTestObj[percentValue] = xTest
  yTrainObj[percentValue] = yTrain
  yTestObj[percentValue] = yTest

def trees_forests(x_train, x_test, y_train, y_test, per):
  # Imputer for handling missing values
  imputer = SimpleImputer(strategy='mean')
  x_train_imputed = imputer.fit_transform(x_train)
  x_test_imputed = imputer.transform(x_test)

  # Create and train the decision tree classifier
  clf = DecisionTreeClassifier(max_depth=5, random_state=777)
  clf.fit(x_train_imputed, y_train)

  # Make predictions and calculate accuracy
  predictions = clf.predict(x_test_imputed)
  accuracy = accuracy_score(y_test, predictions)
  accuracyDT.append(accuracy)

  print(f"Decision Tree Accuracy: {accuracy}")
  if per == 0.1:
    print('\nFeature Importance for 10% Missing values')
    importances = clf.feature_importances_
    for i, importance in enumerate(importances):
      print(f"Feature {x_train.columns[i]}: Importance {importance}")
    print('----------------------------------------------------------------\n')


def custom_random_forest(x_train, x_test, y_train, y_test, per, n_estimators, max_depth):

  label_encoder = LabelEncoder()
  y_train_encoded = label_encoder.fit_transform(y_train)
  y_test_encoded = label_encoder.transform(y_test)

  # Imputer for handling missing values
  imputer = SimpleImputer(strategy='mean')
  x_train_imputed = imputer.fit_transform(x_train)
  x_test_imputed = imputer.transform(x_test)

  # Initialize a list to store individual tree estimators
  trees = []

  feature_importances = np.zeros(x_train.shape[1])  # Array to store feature importances

  # Train individual decision trees
  for _ in range(n_estimators):
    selected_features = np.random.choice(x_train.columns, 5, replace=False)
    selected_indices = [list(x_train.columns).index(feat) for feat in selected_features]

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=None)  # Random state None for diversity
    clf.fit(x_train_imputed[:, selected_indices], y_train_encoded)
    trees.append((clf, selected_indices))

    importances = clf.feature_importances_
    for idx, imp in zip(selected_indices, importances):
      feature_importances[idx] += imp / n_estimators  

  predictions = []
  for tree, features in trees:
    predictions.append(tree.predict(x_test_imputed[:, features]))
  predictions = np.array(predictions)

  # Majority vote
  final_predictions = np.round(np.mean(predictions, axis=0)).astype(int)

  final_predictions = label_encoder.inverse_transform(final_predictions)

  # Calculate accuracy
  accuracy = accuracy_score(y_test, final_predictions)
  accuracyRF.append(accuracy)
  print(f"Random Forest Accuracy: {accuracy}")
  if per == 0.1:
    for i, importance in enumerate(feature_importances):
        print(f"Feature {x_train.columns[i]}: Importance {importance}")

  print('---------------------------------------------------')

accuracyDT = []
accuracyRF = []

for per in xTrainObj:
  print(f'Missing percent: {per * 100}%')
  trees_forests(xTrainObj[per], xTestObj[per], yTrainObj[per], yTestObj[per], per)
  custom_random_forest(xTrainObj[per], xTestObj[per], yTrainObj[per], yTestObj[per], per, n_estimators=100, max_depth=3)

def diagramPlots():
  x = range(len(accuracyDT))
  tick_labels = [f"{int(10 * i)}%" for i in range(len(accuracyDT))]

  # Plotting
  plt.figure(figsize=(10, 5))
  plt.bar(x, accuracyDT, width=0.4, label='Decision Tree', align='center')
  plt.bar(x, accuracyRF, width=0.4, label='Random Forest', align='edge')
  
  plt.xlabel('Missing Percent')
  plt.ylabel('Accuracy')
  plt.title('Comparison of Model Accuracies')
  plt.legend()
  plt.xticks(x, tick_labels)
  plt.show()

diagramPlots()