import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances
from sklearn.metrics.cluster import rand_score

warnings.simplefilter(action='ignore', category=FutureWarning)

file_path = "seeds_dataset.json"
file_path2 = "seeds_dataset_normalized.json"
file_path3 = "seeds_dataset_normalized_sq_eu.json"

seedsDataset = pd.read_json(file_path)                                  # NO normalized data
seedsDataset = pd.DataFrame(seedsDataset['data'].tolist())

seedsDataset_norm = pd.read_json(file_path2)                            # Normalized data Cosine
seedsDataset_norm = pd.DataFrame(seedsDataset_norm['data'].tolist())

seedsDataset_norm_sq = pd.read_json(file_path3)                            # Normalized data squared
seedsDataset_norm_sq = pd.DataFrame(seedsDataset_norm_sq['data'].tolist())

for col in seedsDataset.columns[:-1]:
  seedsDataset[col] = pd.to_numeric(seedsDataset[col], errors='coerce')

distances = pd.DataFrame(distance.cdist(seedsDataset.iloc[:, :-1], seedsDataset.iloc[:, :-1], 'euclidean'))

print(distances)

distancesCosine = pd.DataFrame(distance.cdist(seedsDataset.iloc[:, :-1], seedsDataset.iloc[:, :-1], 'cosine'))

# print(distancesCosine)

# label_counts = seedsDataset['Label'].value_counts()

def calculate_smallest_subset(distances, start_row, end_row, start_col, end_col):
  subset = distances.iloc[start_row:end_row, start_col:end_col]
  series_data = subset.unstack()
  result = series_data.nsmallest(10).unique()

  return result

smallest12 = calculate_smallest_subset(distances, 0, 70, 70, 140)
smallest23 = calculate_smallest_subset(distances, 70, 140, 140, 210)
smallest13 = calculate_smallest_subset(distances, 0, 70, 140, 210)

smallest12Cosine = calculate_smallest_subset(distancesCosine, 0, 70, 70, 140)
smallest23Cosine = calculate_smallest_subset(distancesCosine, 70, 140, 140, 210)
smallest13Cosine = calculate_smallest_subset(distancesCosine, 0, 70, 140, 210)

print('Classes 1 and 2')
print(smallest12)
print('Classes 2 and 3')
print(smallest23)
print('Classes 1 and 3')
print(smallest13)
print('\n')

print('Classes 1 and 2 Cosine')
print(smallest12Cosine)
print('Classes 2 and 3 Cosine')
print(smallest23Cosine)
print('Classes 1 and 3 Cosine')
print(smallest13Cosine)
print('\n')

# print(label_counts)
# print(distances)

'''
We calculated all the distances using 'distances'. This will give us a dataFrame with the diagonal being zeros and top_half = bottom_half (diagonally)
Then we gather top 10 smallest distances between each class (each class will be a subset of distances dataFrame)
We assume that class 2 and 3 are more separable since euclidean and cosine distances between them show to be the biggest numbers 
'''

'''
Silhouette Scores
'''

def calculate_silhouette_score(dataset, n_clusters, random_state=0):
  kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
  labels = kmeans.fit_predict(dataset)
  silhouette_avg = silhouette_score(dataset, labels)                    # metric='euclidean'
  return silhouette_avg, labels

def calculate_silhouette_score_metrCosine(dataset, n_clusters, random_state=0):
  kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
  labels = kmeans.fit_predict(dataset)
  silhouette_avg = silhouette_score(dataset, labels, metric='cosine')   # metric='cosine'
  return silhouette_avg, labels

def predict_labels(dataset, random_state):
  kmeans = KMeans(n_clusters=3, random_state=random_state)
  labels = kmeans.fit_predict(dataset)
  return labels

'''
Plots for metric='euclidean' and metric='cosine'
'''
def plot_silhouette_diagrams(data, cluster_nums):                       # metric='euclidean'
  num_plots = len(cluster_nums)
  fig, axs = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))

  figure_title = 'No Normalization Silhouette Diagram'
  fig.suptitle(figure_title, fontsize=16)  

  for i, n_clusters in enumerate(cluster_nums):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(data)

    silhouette_avg = silhouette_score(data, cluster_labels)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    y_lower = 10
    for j in range(n_clusters):
      ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == j]
      ith_cluster_silhouette_values.sort()

      size_cluster_j = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_j

      color = plt.cm.nipy_spectral(float(j) / n_clusters)
      axs[i].fill_betweenx(np.arange(y_lower, y_upper),
                           0, ith_cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.7)

      axs[i].text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))
      y_lower = y_upper + 10

    axs[i].set_title(f"Silhouette plot for {n_clusters} clusters")
    axs[i].set_xlabel("Silhouette coefficient values")
    axs[i].set_ylabel("Cluster label")

    axs[i].axvline(x=silhouette_avg, color="red", linestyle="--")

    axs[i].set_yticks([])
    axs[i].set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

  plt.show()

def plot_silhouette_diagrams_cosine(data, cluster_nums):                # metric='cosine'
  num_plots = len(cluster_nums)
  fig, axs = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))

  figure_title = 'Normalization Silhouette Diagram Using Cosine Metrics'
  fig.suptitle(figure_title, fontsize=16)  

  for i, n_clusters in enumerate(cluster_nums):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(data)

    silhouette_avg = silhouette_score(data, cluster_labels, metric='cosine')
    sample_silhouette_values = silhouette_samples(data, cluster_labels, metric='cosine')

    y_lower = 10
    for j in range(n_clusters):
      ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == j]
      ith_cluster_silhouette_values.sort()

      size_cluster_j = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_j

      color = plt.cm.nipy_spectral(float(j) / n_clusters)
      axs[i].fill_betweenx(np.arange(y_lower, y_upper),
                           0, ith_cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.7)

      axs[i].text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))
      y_lower = y_upper + 10

    axs[i].set_title(f"Silhouette plot for {n_clusters} clusters")
    axs[i].set_xlabel("Silhouette coefficient values")
    axs[i].set_ylabel("Cluster label")

    axs[i].axvline(x=silhouette_avg, color="red", linestyle="--")

    axs[i].set_yticks([])
    axs[i].set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

  plt.show()

'''
We run simulation of 9 times with n_clusters ranged [2:11]. We gathered 9 different silhouetteAvg and we found the best one (closer to 1).
The best one with NO normalization using euclidean metrics seems to be 2 although 3 is close enough.
If we normalize the data we get the best option to be 3 but silhouetteAvg is lower than before
'''
silhouetteAvgObject = {}                                                # Euclidean NO normalization

for num in range(2,11):
  silhouette_avg, _ = calculate_silhouette_score(seedsDataset, num)     
  silhouetteAvgObject[num] = silhouette_avg

best_num_clusters = max(silhouetteAvgObject, key=silhouetteAvgObject.get)
best_silhouette_avg = silhouetteAvgObject[best_num_clusters]

print(f"The best number of clusters with NO normalization and euclidean metrics is {best_num_clusters} with a silhouette average of {best_silhouette_avg}.")
plot_silhouette_diagrams(seedsDataset, cluster_nums=[2, 3])

'''
We do the same from [2:11].
The best one with normalization using cosine metrics seems to be 2.
If we get 3 classes then one class is half negative half positive with some zeros.
Zeros indicate that classes are overlapping.
Negative values indicate that there is high chance that some samples have been assigned to the wrong cluster, as a different cluster is more similar.
'''
silhouetteAvgObject_norm = {}                                           # cosine WITH normalization

for num in range(2,11):
  silhouette_avg_norm, _ = calculate_silhouette_score_metrCosine(seedsDataset_norm, num)
  silhouetteAvgObject_norm[num] = silhouette_avg_norm

best_num_clusters_norm = max(silhouetteAvgObject_norm, key=silhouetteAvgObject_norm.get)
best_silhouette_avg_norm = silhouetteAvgObject_norm[best_num_clusters_norm]

print(f"The best number of clusters with normalization and cosine metrics is {best_num_clusters_norm} with a silhouette average of {best_silhouette_avg_norm}.")
plot_silhouette_diagrams_cosine(seedsDataset_norm, cluster_nums=[2, 3])

'''

'''

trueLabels = seedsDataset.iloc[:, -1]

predictedLabels = []
randIndexList = []

# squaredEuSeedsDataset = pairwise_distances(seedsDataset, metric='sqeuclidean')          # DO NOT USE
# print('------------------------------------------------\n', squaredEuSeedsDataset)

for num in range(5):
  predictedLabels.append(predict_labels(seedsDataset_norm_sq, num))       # Squared euclidean

for idx, labels in enumerate(predictedLabels):
  ri = rand_score(trueLabels, labels)
  randIndexList.append(ri)
  print(f"Rand Index for prediction with random state {idx}: {ri}")

mean_ri = np.mean(randIndexList)
variance_ri = np.var(randIndexList)

print(f"Mean Rand Index: {mean_ri}")
print(f"Variance of Rand Index: {variance_ri}")

predictedLabels2 = []
randIndexList2 = []

for num in range(5):
  predictedLabels2.append(predict_labels(seedsDataset_norm, num))        # Cosine

for idx, labels in enumerate(predictedLabels2):
  ri = rand_score(trueLabels, labels)
  randIndexList2.append(ri)
  print(f"Rand Index for prediction with random state {idx}: {ri}")

mean_ri2 = np.mean(randIndexList2)
variance_ri2 = np.var(randIndexList2)

print(f"Mean Rand Index: {mean_ri2}")
print(f"Variance of Rand Index: {variance_ri2}")