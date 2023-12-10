import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_dataset(dataset):
  for col in dataset.columns[:-1]:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

  dataset = dataset.dropna()

  labels = dataset["Label"]
  dataset = dataset.drop("Label", axis=1)

  squared_dataset  = np.round(dataset.apply(lambda x: x**2), 8)

  if "Label" in dataset.columns:
    squared_dataset["Label"] = labels

  return squared_dataset

file_path = "../seeds_dataset.json"                                    # Tree has changed
seedsDataset = pd.read_json(file_path)
seedsDataset = pd.DataFrame(seedsDataset['data'].tolist())

preprocessed_seeds_dataset = preprocess_dataset(seedsDataset)
# print(preprocessed_seeds_dataset)
# print(seedsDataset)

jsonDataNoLabel = pd.DataFrame(preprocessed_seeds_dataset.to_dict(orient='records'))
jsonData = pd.concat([jsonDataNoLabel, seedsDataset[["Label"]]], axis=1)
jsonData["Label"] = pd.to_numeric(jsonData["Label"], errors='coerce')

data_as_dict = jsonData.to_dict(orient='records')
json_data = {'data': data_as_dict}

with open('seeds_dataset_normalized_sq_eu.json', 'w') as json_file:
  json.dump(json_data, json_file, indent=2)