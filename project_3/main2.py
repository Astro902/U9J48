import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import rand_score

file_path = "seeds_dataset.json"
file_path2 = "seeds_dataset_normalized.json"
file_path3 = "seeds_dataset_normalized_sq_eu.json"

seedsDataset = pd.read_json(file_path)                                  # NO normalized data
seedsDataset = pd.DataFrame(seedsDataset['data'].tolist())

seedsDataset_norm = pd.read_json(file_path2)                            # Normalized data cosine
seedsDataset_norm = pd.DataFrame(seedsDataset_norm['data'].tolist())

seedsDataset_norm_sq = pd.read_json(file_path3)                         # Normalized data squared
seedsDataset_norm_sq = pd.DataFrame(seedsDataset_norm_sq['data'].tolist())

linkage_methods = ['ward', 'complete', 'average', 'single']
selectMethod = 2

true_labels = seedsDataset.iloc[:, -1].copy()

# print(true_labels)
# print(seedsDataset)

def AGLON(data, method):
  agg_cluster = AgglomerativeClustering(n_clusters=3, linkage=method, metric='manhattan')
  agg_labels = agg_cluster.fit_predict(data)
  return agg_labels

def agDendrogram(data, method):
  linkage_matrix = linkage(data, method=method, metric='cityblock')
  dendrogram(linkage_matrix)
  plt.show()

predicted_labels = AGLON(seedsDataset, linkage_methods[selectMethod])
# print(predicted_labels)

ri = rand_score(true_labels, predicted_labels)
print(f'Rand Index: {ri}')

# ari = adjusted_rand_score(true_labels, predicted_labels)
# print(f'Adjusted Rand Index: {ari}')

dendrogramVisualize = agDendrogram(seedsDataset, linkage_methods[selectMethod])
