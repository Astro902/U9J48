from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.layers import Layer, Conv2D, MaxPooling2D, concatenate, Flatten, Dense, Input

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix


import warnings
warnings.filterwarnings('ignore')

dataGen = ImageDataGenerator(
  rescale=1./255,
)

trainGen = dataGen.flow_from_directory(
  'splitted_dataset/train',
  target_size=(28, 28),
  batch_size=32,
  class_mode='binary',  # 'binary' for two classes, 'categorical' for more
)
validGen = dataGen.flow_from_directory(
  'splitted_dataset/validation',
  target_size=(28, 28),
  batch_size=32,
  class_mode='binary',  # 'binary' for two classes, 'categorical' for more
)
testGen = dataGen.flow_from_directory(
  'splitted_dataset/test',
  target_size=(28, 28),
  batch_size=32,
  class_mode='binary',  # 'binary' for two classes, 'categorical' for more
)

# print(trainGen.image_shape)
# print(validGen.image_shape)
# print(testGen.image_shape)

class InceptionBlock(Layer):
  def __init__(self, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    super(InceptionBlock, self).__init__()
    
    # 1x1 conv branch
    self.conv1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')

    # 3x3 conv branch
    self.conv3x3_reduce = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')
    self.conv3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')

    # 5x5 conv branch
    self.conv5x5_reduce = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')
    self.conv5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')

    # Pooling branch
    self.pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    self.pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')

  def call(self, inputs):
    # Concatenate all branches
    branch1 = self.conv1x1(inputs)
    
    branch2 = self.conv3x3_reduce(inputs)
    branch2 = self.conv3x3(branch2)
    
    branch3 = self.conv5x5_reduce(inputs)
    branch3 = self.conv5x5(branch3)
    
    branch4 = self.pool(inputs)
    branch4 = self.pool_proj(branch4)
    
    return concatenate([branch1, branch2, branch3, branch4], axis=-1)

class MyModel(keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()

    self.conv7x7 = Conv2D(filters=64, kernel_size=7, strides=2, padding='same')

    self.block1 = InceptionBlock(filters_1x1=8, filters_3x3=12, filters_3x3_reduce=8, filters_5x5=12, filters_5x5_reduce=8, filters_pool_proj=12)

    self.flatten = Flatten()
    self.fc = Dense(128, activation='relu')
    self.out = Dense(1, activation='sigmoid')  # or 'softmax' for multiclass

  def call(self, inputs):
    x = self.conv7x7(inputs)
    x = self.block1(x)

    x = self.flatten(x)
    x = self.fc(x)
    x = self.out(x)
    return x
  
  def summary(self):
    x = Input(shape=(24, 24, 3))
    model = keras.Model(inputs=[x], outputs=self.call(x))
    return model.summary()

if __name__ == '__main__':
  sub = MyModel()
  sub.summary()

# Instantiate the model
model = MyModel()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
  trainGen,
  steps_per_epoch=trainGen.samples // trainGen.batch_size,
  epochs=10,
  validation_data=validGen,
  validation_steps=validGen.samples // validGen.batch_size,
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(
  testGen,
  steps=testGen.samples // testGen.batch_size,
)

print(f"Test accuracy: {test_accuracy}")

def get_predictions(model, generator):
  generator.reset()
  predictions = model.predict(generator, steps=np.ceil(generator.samples / generator.batch_size))
  predicted_labels = (predictions > 0.5).astype(int).flatten()
  true_labels = generator.classes
  return predicted_labels, true_labels

def calculate_metrics(predicted_labels, true_labels):
  precision = precision_score(true_labels, predicted_labels)
  recall = recall_score(true_labels, predicted_labels)
  auc = roc_auc_score(true_labels, predicted_labels)

  print(f"Precision: {precision:.4f}")
  print(f"Recall: {recall:.4f}")
  print(f"AUC: {auc:.4f}")

predicted_labels, true_labels = get_predictions(model, testGen)

calculate_metrics(predicted_labels, true_labels)