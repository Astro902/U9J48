import pandas as pd
import numpy as np
import warnings
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

file_path = "seeds_dataset.json"
file_path2 = "seeds_dataset_normalized.json"

seedsDataset = pd.read_json(file_path)
seedsDataset = pd.DataFrame(seedsDataset['data'].tolist())

seedsDataset_norm = pd.read_json(file_path2)
seedsDataset_norm = pd.DataFrame(seedsDataset_norm['data'].tolist())

print(seedsDataset_norm)

for col in seedsDataset.columns[:-1]:
  seedsDataset[col] = pd.to_numeric(seedsDataset[col], errors='coerce')

distances = pd.DataFrame(distance.cdist(seedsDataset.iloc[:, :-1], seedsDataset.iloc[:, :-1], 'euclidean'))
distancesCosine = pd.DataFrame(distance.cdist(seedsDataset.iloc[:, :-1], seedsDataset.iloc[:, :-1], 'cosine'))
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

def calculate_silhouette_score(dataset, n_clusters, random_state=0):
  kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
  labels = kmeans.fit_predict(dataset)
  silhouette_avg = silhouette_score(dataset, labels)
  return silhouette_avg, labels

def calculate_silhouette_score_metrCosine(dataset, n_clusters, random_state=0):
  kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
  labels = kmeans.fit_predict(dataset)
  silhouette_avg = silhouette_score(dataset, labels, metric='cosine')
  return silhouette_avg, labels

silhouetteAvgObject = {} # Euclidean NO normalization

for num in range(2,11):
  silhouette_avg, _ = calculate_silhouette_score(seedsDataset, num)
  silhouetteAvgObject[num] = silhouette_avg

best_num_clusters = max(silhouetteAvgObject, key=silhouetteAvgObject.get)
best_silhouette_avg = silhouetteAvgObject[best_num_clusters]

print(f"The best number of clusters is {best_num_clusters} with a silhouette average of {best_silhouette_avg}.")

'''
We run simulation of 9 times with n_clusters ranged [2:11]. We gathered 9 different silhouetteAvg and we found the best one closer to 1.
The best one seems to be 2 although 3 is close enough.
'''

def plot_silhouette_diagrams(data, cluster_nums):
  num_plots = len(cluster_nums)
  fig, axs = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))

  figure_title = 'No Normalization Silhouette Diagram'
  fig.suptitle(figure_title, fontsize=16)  

  for i, n_clusters in enumerate(cluster_nums):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
    cluster_labels = clusterer.fit_predict(data)

    silhouette_avg = silhouette_score(data, cluster_labels)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
      ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
      ith_cluster_silhouette_values.sort()

      size_cluster_i = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_i

      color = cm.nipy_spectral(float(i) / n_clusters)
      ax1.fill_betweenx(
          np.arange(y_lower, y_upper),
          0,
          ith_cluster_silhouette_values,
          facecolor=color,
          edgecolor=color,
          alpha=0.7,
        )

      ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
      y_lower = y_upper + 10

    ax1.set_title(f"Silhouette plot for {n_clusters} clusters")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

    npData = data.values

    ax2.scatter(
      npData[:, 0], npData[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    centers = clusterer.cluster_centers_

    ax2.scatter(
      centers[:, 0],
      centers[:, 1],
      marker="o",
      c="white",
      alpha=1,
      s=200,
      edgecolor="k",
    )

    for i, c in enumerate(centers):
      ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
      "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
      % n_clusters,
      fontsize=14,
      fontweight="bold",
    )

  plt.show()
range_n_clusters = [2, 3, 4]
plot_silhouette_diagrams(seedsDataset, range_n_clusters)

'''
Plots
'''

silhouetteAvgObject_norm = {} # cosine WITH normalization

for num in range(2,11):
  silhouette_avg_norm, _ = calculate_silhouette_score_metrCosine(seedsDataset_norm, num)
  silhouetteAvgObject_norm[num] = silhouette_avg_norm

best_num_clusters_norm = max(silhouetteAvgObject_norm, key=silhouetteAvgObject_norm.get)
best_silhouette_avg_norm = silhouetteAvgObject_norm[best_num_clusters_norm]

print(f"The best number of clusters is {best_num_clusters_norm} with a silhouette average of {best_silhouette_avg_norm}.")

plot_silhouette_diagrams(seedsDataset_norm, range_n_clusters)