import pandas as pd
import re
import json

# Open the text file and read its content
with open('seeds_dataset.txt', 'r') as file:                                    # Tree has changed
  content = file.read()

# Extract numbers using regular expression
numbers = re.findall(r'\b\d+\.\d+|\b\d+\b', content)
print(numbers)

## Define the number of elements per row
elements_per_row = 8

# Create a list of rows, each containing the specified number of elements
rows = [numbers[i:i + elements_per_row] for i in range(0, len(numbers), elements_per_row)]

# Complicated af
df = pd.DataFrame(rows, columns=([f'Attribute{i+1}' for i in range(elements_per_row - 1)] + ['Label']))

data_as_dict = df.to_dict(orient='records')
json_data = {'data': data_as_dict}

with open('seeds_dataset.json', 'w') as json_file:
  json.dump(json_data, json_file, indent=2)

# print(df)