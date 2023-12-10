import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

file_path = "seeds_dataset.json"

seedsDataset = pd.read_json(file_path)
seedsDataset = pd.DataFrame(seedsDataset['data'].tolist())

print(seedsDataset)

linkage_methods = ['ward', 'complete', 'average', 'single']

# Create subplots for each linkage method
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, linkage_method in enumerate(linkage_methods):
  # Perform agglomerative clustering
  agg_cluster = AgglomerativeClustering(n_clusters=4, linkage=linkage_method)
  agg_labels = agg_cluster.fit_predict(seedsDataset)

  # Plot the data points with cluster assignments
  axes[i].scatter(seedsDataset.iloc[:, 0], seedsDataset.iloc[:, 1], c=agg_labels, cmap='viridis')
  axes[i].set_title(f'Agglomerative Clustering ({linkage_method} linkage)')
  axes[i].set_xlabel('Feature 1')
  axes[i].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()