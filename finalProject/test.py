import cv2
import os
import numpy as np
import keras

incdir = 'C:/ProjectIgnis/.py/data/mask_incorrect_use'

newInputs = keras.utils.image_dataset_from_directory(
  directory=incdir,
  labels=None,
  image_size=(28, 28),
  seed=777,
  batch_size=32,
  shuffle=True,
)

newInputs = newInputs.map(lambda x: (x/255))
newInputs.as_numpy_iterator()

Test_Inputs = newInputs.take(len(newInputs))

for batch in Test_Inputs.as_numpy_iterator():
  for picture in batch:
    print(picture)
    print('----------------------------------------------------')