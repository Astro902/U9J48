import os
import shutil
from sklearn.model_selection import train_test_split

# Function to split data
def split_data(source, train_dir, val_dir, test_dir, train_size=0.6):
  files = [os.path.join(source, f) for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
  train_files, test_files = train_test_split(files, train_size=train_size, random_state=777)
  val_files, test_files = train_test_split(test_files, train_size=0.5, random_state=777)

  for file in train_files:
    shutil.move(file, train_dir)
  for file in val_files:
    shutil.move(file, val_dir)
  for file in test_files:
    shutil.move(file, test_dir)

# Directories for the dataset
original_dir = 'Data/Mask_DB'
target_dir = 'splitted_dataset'
classes = ['with_mask', 'without_mask']
sets = ['train', 'validation', 'test']

if not os.path.exists(target_dir):
  os.mkdir(target_dir)

# Create new directories
for set_name in sets:
  set_dir = os.path.join(target_dir, set_name)
  if not os.path.exists(set_dir):
    os.mkdir(set_dir)
  for class_name in classes:
    class_dir = os.path.join(set_dir, class_name)
    if not os.path.exists(class_dir):
      os.mkdir(class_dir)

# Split data for each class
for class_name in classes:
  source_dir = os.path.join(original_dir, class_name)
  train_dir = os.path.join(target_dir, 'train', class_name)
  val_dir = os.path.join(target_dir, 'validation', class_name)
  test_dir = os.path.join(target_dir, 'test', class_name)

  if not os.listdir(train_dir) and not os.listdir(val_dir) and not os.listdir(test_dir):  # Check if directories are empty
    split_data(source_dir, train_dir, val_dir, test_dir)
  else:
    print(f"Data already split for class {class_name}")
