import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D

warnings.simplefilter(action='ignore', category=FutureWarning)

file_path1 = "allTXTfiles/Air_Distances_Between_Cities_in_Statute_Miles/City_names_US.txt"
file_path2 = "allTXTfiles/Air_Distances_Between_Cities_in_Statute_Miles/City_names_world.txt"
file_path3 = "allTXTfiles/Air_Distances_Between_Cities_in_Statute_Miles/Distance_Matrix_US.txt"
file_path4 = "allTXTfiles/Air_Distances_Between_Cities_in_Statute_Miles/Distance_Matrix_world.txt"

citiesUS_table = pd.read_csv(file_path1, header=None)
citiesWorld_table = pd.read_csv(file_path2, header=None)
dissimilarityMatrix_US = pd.read_csv(file_path3, delimiter=' ', header=None)
dissimilarityMatrix_world = pd.read_csv(file_path4, delimiter=' ', header=None)

dissimilarityMatrix_US.columns = citiesUS_table[0].tolist()
dissimilarityMatrix_world.columns = citiesWorld_table[0].tolist()

# print(dissimilarityMatrix_US)
# print(dissimilarityMatrix_world)


def mdsDiagram(dissimilarity_matrix):
  mds = MDS(n_components=2, n_init=100, max_iter=10000, dissimilarity='precomputed', random_state=0)
  mds_coordinates_initial = mds.fit_transform(dissimilarity_matrix)

  # Print the initial coordinates
  print("Initial MDS Coordinates:")
  print(mds_coordinates_initial)

  # Step 2: Second MDS computation using the initial embedding
  nmds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
  mds_coordinates_refined = nmds.fit_transform(dissimilarity_matrix, init=mds_coordinates_initial)

  # Print the refined coordinates
  print("\nRefined MDS Coordinates:")
  print(mds_coordinates_refined)

  # Step 3: Create a Plot with the refined coordinates
  plt.figure(figsize=(10.5, 7.5))
  plt.scatter(mds_coordinates_refined[:, 0], -mds_coordinates_refined[:, 1])

  # Add labels to the points
  for i, txt in enumerate(dissimilarity_matrix.columns):
    plt.annotate(txt, (mds_coordinates_refined[i, 0], -mds_coordinates_refined[i, 1]))

  plt.title('Refined MDS Plot')
  plt.xlabel('MDS Dimension 1')
  plt.ylabel('MDS Dimension 2')
  plt.show()

def mdsDiagram3d(dissimilarity_matrix):
  # Step 1: First MDS computation
  mds = MDS(n_components=3, n_init=100, max_iter=10000, dissimilarity='precomputed', random_state=0)
  mds_coordinates_initial = mds.fit_transform(dissimilarity_matrix)

  # Step 2: Second MDS computation using the initial embedding
  nmds = MDS(n_components=3, dissimilarity='precomputed', random_state=0)
  mds_coordinates_refined = nmds.fit_transform(dissimilarity_matrix, init=mds_coordinates_initial)

  # Step 3: Create a Plot with the refined coordinates
  fig = plt.figure(figsize=(10.5, 7.5))
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(mds_coordinates_refined[:, 0], -mds_coordinates_refined[:, 1], -mds_coordinates_refined[:, 2])

  # Add labels to the points
  for i, txt in enumerate(dissimilarity_matrix.columns):
    ax.text(mds_coordinates_refined[i, 0], -mds_coordinates_refined[i, 1], -mds_coordinates_refined[i, 2], txt)

  ax.set_title('3D MDS Plot')
  ax.set_xlabel('MDS Dimension 1')
  ax.set_ylabel('MDS Dimension 2')
  ax.set_zlabel('MDS Dimension 3')

  plt.show()

def yyt(dissimilarity_matrix):
  # Step 2: Apply MDS Algorithm
  mds = MDS(n_components=2, dissimilarity='precomputed')
  cities_coordinates = mds.fit_transform(dissimilarity_matrix)

  # Step 3: Table with Vector Representations
  vector_representations = pd.DataFrame(cities_coordinates, index=dissimilarity_matrix.index)

  print(vector_representations)

  # Step 4: Eigenvalues of Y @ Y(transpose)
  eigenvalues, _ = np.linalg.eig(np.dot(cities_coordinates, cities_coordinates.T))

  # Step 5: Create Eigenvalues Chart
  plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o')
  plt.xlabel('Dimension')
  plt.ylabel('Eigenvalue')
  plt.title('Eigenvalues of Y @ Y(transpose)')
  plt.show()

  
# plotUS = mdsDiagram(dissimilarityMatrix_US)
plotWorld = mdsDiagram(dissimilarityMatrix_world)

# plotUS3d = mdsDiagram3d(dissimilarityMatrix_US)
plotWorld3d = mdsDiagram3d(dissimilarityMatrix_world)

# qwerty = yyt(dissimilarityMatrix_world)
# qwerty = yyt(dissimilarityMatrix_US)