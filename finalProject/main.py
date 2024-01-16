import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Layer, concatenate, BatchNormalization
from keras.metrics import Precision, Recall, BinaryAccuracy, AUC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

file_dir = 'C:/ProjectIgnis/.py/data/Mask_DB'

incdir0 = 'C:/ProjectIgnis/.py/data/Mask_DB/with_mask'
incdir1 = 'C:/ProjectIgnis/.py/data/Mask_DB/without_mask'
incdir404 = 'C:/ProjectIgnis/.py/data/mask_incorrect_use'

data = tf.keras.utils.image_dataset_from_directory(
  directory=file_dir,                                   # ----------- Reminder ------------ #
  image_size=(32, 32),                                  # Class0 = WITH Mask                #
  seed=777,                                             # Class1 = NO Mask                  #  
  batch_size=32,                                        # CLass404 = Incorrect Use of Mask  #
  shuffle=True,                                         # --------------------------------- #
) 

def loadData(dataDir):
  data = tf.keras.utils.image_dataset_from_directory(
    directory=dataDir,
    labels=None,
    image_size=(32, 32),
    batch_size=32,
    shuffle=False,
  )
  data = data.map(lambda x: (x/255))
  data.as_numpy_iterator()
  TestData = data.take(len(data))
  return TestData

data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator()

train_size = round(len(data) * .6)
val_size = round(len(data) * .2)
test_size = round(len(data) * .2)

while train_size + val_size + test_size < len(data):
  train_size += 1

while train_size + val_size + test_size > len(data):
  train_size -= 1
  
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

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

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()

    self.conv5x5_1 = Conv2D(filters=32, kernel_size=5, strides=1, padding='same')
    self.conv3x3_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
    self.conv3x3_3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')

    self.batchNorm = BatchNormalization()
    self.maxPool = MaxPooling2D()
    self.dropout = Dropout(0.1)

    self.block1 = InceptionBlock(filters_1x1=8, filters_3x3=16, filters_3x3_reduce=8, filters_5x5=16, filters_5x5_reduce=8, filters_pool_proj=16)

    self.flatten = Flatten()
    self.fc1 = Dense(128, activation='relu')
    self.out = Dense(1, activation='sigmoid')

  def call(self, inputs):
    x = self.conv5x5_1(inputs)
    x = self.batchNorm(x)
    x = self.maxPool(x)

    x = self.conv3x3_2(x)
    x = self.maxPool(x)

    x = self.conv3x3_3(x)

    # x = self.block1(x)

    x = self.flatten(x)
    x = self.fc1(x)
    x = self.dropout(x)
    x = self.out(x)
    return x
  
  def summary(self):
    x = Input(shape=(32, 32, 3))
    model = Model(inputs=[x], outputs=self.call(x))
    return model.summary()

if __name__ == '__main__':
  sub = MyModel()
  sub.summary()

model = MyModel()

model.compile(optimizer='Adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

logdir = 'C:/ProjectIgnis/.py/logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

history = model.fit(
  train,
  epochs=10,
  validation_data=val,
  callbacks=[tensorboard_callback]
)

fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='Loss')
plt.plot(history.history['val_loss'], color='salmon', label='Valodation Loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper right')
plt.show()

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
  X, y = batch
  yhat = model.predict(X)
  pre.update_state(y, yhat)
  re.update_state(y, yhat)
  acc.update_state(y, yhat)

print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Acc:{acc.result().numpy()}')

testInput0 = loadData(incdir0)
testInput1 = loadData(incdir1)
testInput404 = loadData(incdir404)

def showPredict(inputs):
  preds = []
  for batch in inputs.as_numpy_iterator():
    yhat = model.predict(batch)
    for item in yhat:
      if item >= .008:
        item = 1
      if item < .008:
        item = 0
      # else: print('How did you get here')
      preds.append(item)
  
  return preds

predsMask = showPredict(testInput0)
predsNoMask = showPredict(testInput1)
predsIncMask = showPredict(testInput404)

def predPrint(preds):
  if preds == predsMask:
    print('-------- Predicted WITH Mask --------')
  if preds == predsNoMask:
    print('-------- Predicted NO Mask --------')
  if preds == predsIncMask:
    print('-------- Predicted INCORRECT USE Mask --------')

  print('People with mask: ', preds.count(0))
  print('People without mask: ', preds.count(1))
  print('Accuracy Incorrect Use: ', preds.count(1) / (preds.count(0) + preds.count(1)))
  
predPrint(predsMask)
predPrint(predsNoMask)
predPrint(predsIncMask)

zeros_list = [1] * len(predsIncMask)

cm = confusion_matrix(zeros_list, predsIncMask)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()