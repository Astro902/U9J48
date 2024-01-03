import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, auc

class IrisModel(tf.keras.Model):
  def __init__(self, num_classes):
    super(IrisModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(30, activation='ReLU')                                
    self.dense2 = tf.keras.layers.Dense(20, activation='ReLU')                                #C We add 3rd Layer 
    self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(inputs)
    return self.dense(x)
  
def testAccDiagram(testAccuracy):
  plt.plot(range(1, numEpochs + 1), testAccuracy)
  plt.xlabel('Epoch')
  plt.ylabel('Test Accuracy')
  plt.title('Test Accuracy over Epochs')
  plt.show()

def confusionMatrixDiagram(testData, testLabels):
  predictions = model.predict(testData)
  predicted_labels = np.argmax(predictions, axis=1)
  cm = confusion_matrix(testLabels, predicted_labels)
  sns.heatmap(cm, annot=True, fmt='g')
  plt.xlabel('Predicted labels')
  plt.ylabel('True labels')
  plt.title('Confusion Matrix')
  plt.show()
    
def ROCcurveDiagram(X_test, y_test):
  probabilities = model.predict(X_test)
  predicted_labels = np.argmax(probabilities, axis=1)

  # For multi-class, use 'macro', 'micro', or 'weighted'
  precision = precision_score(y_test, predicted_labels, average='macro')
  recall = recall_score(y_test, predicted_labels, average='macro')

  #/ Calculating AUC for each class and averaging
  auc_scores = []
  for i in range(numClasses):
    class_probabilities = probabilities[:, i]
    class_auc = roc_auc_score(y_test == i, class_probabilities)
    auc_scores.append(class_auc)
  avg_auc = np.mean(auc_scores)

  print("Precision:", precision)
  print("Recall:", recall)
  print("Average AUC:", avg_auc)

    # Plotting ROC Curve for each class
  for i in range(numClasses):
    fpr, tpr, _ = roc_curve(y_test == i, probabilities[:, i])
    plt.plot(fpr, tpr, label=f'Class {i} AUC = {auc_scores[i]:.2f}')
  plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve')
  plt.legend()
  plt.show()

def PrecisionRecallCurveDiagram(X_test, y_test, numClasses):
  probabilities = model.predict(X_test)
  precision_dict = {}
  recall_dict = {}
  pr_auc_dict = {}

  # Plotting Precision-Recall Curve for each class
  for i in range(numClasses):
    class_probabilities = probabilities[:, i]
    precision, recall, _ = precision_recall_curve(y_test == i, class_probabilities)
    pr_auc = auc(recall, precision)
    
    precision_dict[i] = precision
    recall_dict[i] = recall
    pr_auc_dict[i] = pr_auc

    plt.plot(recall, precision, label=f'Class {i} AUC = {pr_auc:.2f}')

  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall Curve')
  plt.legend()
  plt.show()

  return precision_dict, recall_dict, pr_auc_dict

numEpochs = 15                                                                                #C We change numEpochs from 10 ---> 15
  
iris = fetch_ucirepo(id=53)                                                                   # fetch dataset 

X = iris.data.features 
y = iris.data.targets 

labelEncoder = LabelEncoder()
yEncoded = labelEncoder.fit_transform(y)

numClasses = len(labelEncoder.classes_)

# print(X.shape) 
# print(y.shape) 
# print(yEncoded)

xTrain, xTest, yTrain, yTest = train_test_split(X, yEncoded, train_size=0.8, random_state=777)

model = IrisModel(numClasses)

#/ compile the model
model.compile(
  optimizer='Adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

# model.fit(xTrain, yTrain, epochs=10, batch_size=32)                                         # Fit model
# test_loss, test_acc = model.evaluate(xTest, yTest)                                          # Evaluate the model on the test set
# print(f'Test accuracy: {test_acc}')

testAccuracy = []
for epoch in range(numEpochs):
  model.fit(xTrain, yTrain, epochs=1, batch_size=8)                                           #C Batch size from 32 ---> 8
  test_loss, test_acc = model.evaluate(xTest, yTest)
  testAccuracy.append(test_acc)

testAcc = testAccDiagram(testAccuracy)
confMatrix = confusionMatrixDiagram(xTest, yTest)
ROCcurve = ROCcurveDiagram(xTest, yTest)
precision_dict, recall_dict, pr_auc_dict = PrecisionRecallCurveDiagram(xTest, yTest, numClasses)
