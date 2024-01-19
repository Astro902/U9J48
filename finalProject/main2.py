import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Layer, concatenate, BatchNormalization
from keras.metrics import Precision, Recall, BinaryAccuracy, AUC

dataFrameFeatures = pd.read_csv('Data/Train_Features.csv')
dataFrameLabels = pd.read_csv('Data/Train_Labels.csv')
dataFrameTest = pd.read_csv('Data/Test_Features.csv')

print(dataFrameFeatures.shape)
print(dataFrameLabels.shape)

pca = PCA(n_components=31)
pca.fit(dataFrameFeatures)
scores = pca.transform(dataFrameFeatures)
scoresDF = pd.DataFrame(scores)

explainedVariance = pca.explained_variance_ratio_

print(explainedVariance)
print(scores.shape)

currentSum = 0
cond1 = False
cond2 = False
cond3 = False
cond4 = False

for i in range(len(explainedVariance)):
  currentSum += explainedVariance[i]
  if currentSum > 0.90 and not cond1:
    print(f'In order to obtain above 90% Reconstruction value there are required {i+1} dimension(s) ({currentSum})')
    cond1 = True
  if currentSum > 0.99 and not cond2:
    print(f'In order to obtain above 99% Reconstruction value there are required {i+1} dimension(s) ({currentSum})')
    cond2 = True
  if currentSum > 0.999 and not cond3:
    print(f'In order to obtain above 99.9% Reconstruction value there are required {i+1} dimension(s) ({currentSum})')
    cond3 = True
  if currentSum > 0.9999 and not cond4:
    print(f'In order to obtain above 99.99% Reconstruction value there are required {i+1} dimension(s) ({currentSum})')
    cond4 = True
  # print(f"Sum of the first {i+1} items:", currentSum)

if isinstance(scores, np.ndarray):
  print("It's a NumPy array")
else:
  print("It's not a NumPy array")

