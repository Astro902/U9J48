import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ------------------------------- #
#                                 #
#       Data construction         #
#                                 #
# ------------------------------- #

file_path = "seeds_dataset.json"
file_path2 = "seeds_dataset_normalized.json"

seedsDataset = pd.read_json(file_path)                                  # NO normalized data
seedsDataset = pd.DataFrame(seedsDataset['data'].tolist())

print(seedsDataset.head())

# seedsDataset_norm = pd.read_json(file_path2)                            # Normalized data
# seedsDataset_norm = pd.DataFrame(seedsDataset_norm['data'].tolist())

for col in seedsDataset.columns[:-1]:
  seedsDataset[col] = pd.to_numeric(seedsDataset[col], errors='coerce')

seedsDatasetLabels = seedsDataset.iloc[:, -1]

# ------------------------------- #
#                                 #
#            Functions            #
#                                 #
# ------------------------------- #

def pcaDecompose(data, n_components):
  pca = PCA(n_components=n_components)
  data_pca = pca.fit_transform(data)                        # principal components
  return pca, data_pca

def pcaDiagram(data, labels):
  plt.figure(figsize=(8, 6))

  for label in set(labels):
    plt.scatter(data[labels == label, 0], data[labels == label, 1], label=f'Class {label}')

  plt.title('PCA')
  plt.xlabel('Principal Component 1 (PC1)')
  plt.ylabel('Principal Component 2 (PC2)')
  plt.show()

def pcaDiagram3d(data, data_pca):
  fig = plt.figure(figsize=(13, 6.5))
    
  ax = fig.add_subplot(121, projection='3d')
  ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c='b', marker='o', label='Original Data')
  ax.set_title('Original Data - First 3 Dimensions')

  ax = fig.add_subplot(122, projection='3d')
  ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c='r', marker='o', label='PCA Data')
  ax.set_title('Data After PCA - 3 Dimensions')

  plt.show()

def ldaDiagram(dataLDA, labels):
  plt.figure(figsize=(8, 6))

  for label in set(labels):
    plt.scatter(dataLDA[labels == label, 0], dataLDA[labels == label, 1], label=f'Class {label}')

  plt.title('LDA - Dimensionality Reduction')
  plt.xlabel('LD1')
  plt.ylabel('LD2')
  plt.legend()
  plt.show()

def diagramScatter(data, labels ,num):
  plt.figure(figsize=(8, 6))

  for label in set(labels):
    plt.scatter(data[labels == label, 0], data[labels == label, 1], label=f'Class {label}')

  if num == 1:
    plt.title('Best 2 features')
    plt.xlabel('3rd feature')
    plt.ylabel('4th feature')
  if num == 2:
    plt.title('Worst 2 features')
    plt.xlabel('5th feature')
    plt.ylabel('6th feature')
  plt.show()

pcaDec, seedsDatasetPCA = pcaDecompose(seedsDataset, 7)

explained_variance_ratio = pcaDec.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# LDA model - fit to entire dataset
lda = LinearDiscriminantAnalysis(n_components=2)
seedsDatasetLDA = lda.fit_transform(seedsDataset, seedsDatasetLabels)
ldaCoefficients = lda.coef_

# ------------------------------- #
#                                 #
#        Prints - Diagrams        #
#                                 #
# ------------------------------- #

print(f'Τα σφάλματα ανακατασκευής των δεδομένων δίνονται από το διάνυσμα: ', explained_variance_ratio)
print(f'Το πόσο κοντά είμαστε από το πραγματικό ειναι: ', cumulative_explained_variance)

# pcaDiagram3d(seedsDataset, seedsDatasetPCA)

pcaDiagram(seedsDatasetPCA, seedsDatasetLabels)

ldaDiagram(seedsDatasetLDA, seedsDatasetLabels)

print('LDA coefficients\n', ldaCoefficients)

best_contributors_features = seedsDataset.iloc[:, [2, 3]].copy()                  # Best 2 according to  LDA Coefficients 
worst_contributors_features = seedsDataset.iloc[:, [4, 5]].copy()                 # Worst 2 according to  LDA Coefficients 
diagramScatter(best_contributors_features.values, seedsDatasetLabels, 1)
diagramScatter(worst_contributors_features.values, seedsDatasetLabels, 2)
