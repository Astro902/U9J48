import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow import keras
from keras.layers import Dense, Flatten, Reshape
from keras.datasets import mnist
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data

class Autoencoder(tf.keras.Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    # Encoder
    self.flatten = Flatten()
    self.encoder1 = Dense(128, activation='relu')
    self.encoder2 = Dense(32, activation='relu')
    self.encoder3 = Dense(3, activation='relu')

    # Decoder
    self.decoder1 = Dense(32, activation='relu')
    self.decoder2 = Dense(128, activation='relu') 
    self.decoder3 = Dense(784, activation='sigmoid')
    self.reshape = Reshape((28, 28))

  def call(self, inputs):
    x = self.flatten(inputs)
    x = self.encoder1(x)
    x = self.encoder2(x)
    x = self.encoder3(x)
    x = self.decoder1(x)
    x = self.decoder2(x)
    x = self.decoder3(x)
    return self.reshape(x)
  
  def encode(self, inputs):
    x = self.flatten(inputs)
    x = self.encoder1(x)
    x = self.encoder2(x)
    return x
  
  def decode(self, encoded):
    x = self.decoder1(encoded)
    x2 = self.decoder2(x)
    decoded = self.decoder3(x2)
    return self.reshape(decoded)

#/ Create an instance of the autoencoder
autoencoder = Autoencoder()

#/ Compile the autoencoder
adam_opt = Adam(learning_rate=0.001)
autoencoder.compile(
  optimizer=adam_opt, loss='binary_crossentropy'
)

# Train the autoencoder
history = autoencoder.fit(
  x_train, x_train,  # Note that both inputs and targets are x_train
  batch_size=32,
  epochs=5,
  validation_data=(x_test, x_test)
)

# Now the autoencoder can be used to encode and decode MNIST images
x_flattened = autoencoder.flatten(x_test)                                     # TEST IMAGES
x_encoded1 = autoencoder.encoder1(x_flattened)
x_encoded2 = autoencoder.encoder2(x_encoded1)
encoded_imgs = autoencoder.encoder3(x_encoded2)

x_decoded1 = autoencoder.decoder1(encoded_imgs)
x_decoded2 = autoencoder.decoder2(x_decoded1)
x_decoded3 = autoencoder.decoder3(x_decoded2)
decoded_imgs = autoencoder.reshape(x_decoded3).numpy()

x_flattened_train = autoencoder.flatten(x_train)                               # TRAIN IMAGES
x_encoded1_train = autoencoder.encoder1(x_flattened_train)
x_encoded2_train = autoencoder.encoder2(x_encoded1_train)
encoded_imgs_train = autoencoder.encoder3(x_encoded2_train)

n = 10  # Number of digits to display

def imageDisplay(n):
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()

def imageDisplayLatentSpace(img, labelColor, indicator):

  fig = plt.figure(figsize=(10, 7))
  ax = fig.add_subplot(111, projection='3d')

  # Extract the values for each dimension
  x_vals = img[:, 0]
  y_vals = img[:, 1]
  z_vals = img[:, 2]

  # Plot the points
  scatter = ax.scatter(x_vals, y_vals, z_vals, c=labelColor, cmap='viridis')

  # Add color bar which maps values to colors.
  cbar = fig.colorbar(scatter, shrink=0.5, aspect=5)
  cbar.set_label('Value in third dimension')

  ax.set_xlabel('Latent X')
  ax.set_ylabel('Latent Y')
  ax.set_zlabel('Latent Z')
  if indicator == '0':
    plt.title('Latent Space 3D Scatter Plot Test Data')
  if indicator == '1':
    plt.title('Latent Space 3D Scatter Plot Train Data')
  else:
    plt.title('Latent Space 3D Scatter Plot')

  plt.show()

def imageGenerator(n):
  latent_dim = 3  # Dimension of the latent space
  # Generate random samples from the latent space
  random_latent_vectors = np.random.normal(size=(n, latent_dim))
  # Decode them to create new images
  generated_images = autoencoder.decode(random_latent_vectors)
  generated_images_np = generated_images.numpy()
  # Visualize the generated images
  plt.figure(figsize=(20, 4))
  for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(generated_images_np[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()

def imageGeneratorTailored():

  tailored_latent_vectors = np.array([
    [20, 30, 10],
    [50, 20, 30],
    [70, 10, 40],
    [30, 40, 30],
    [10, 30, 70]
  ])

  n = 5

  generated_images = autoencoder.decode(tailored_latent_vectors).numpy()

  plt.figure(figsize=(20, 4))
  for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(generated_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()

train_loss_history = []
train_loss_history.extend(history.history['loss'])

squared_sum = sum(loss**2 for loss in train_loss_history)
print('Average squared loss 5 epochs', squared_sum)

imageDisplay(n)
imageDisplayLatentSpace(encoded_imgs_train, y_train, '1')
imageDisplayLatentSpace(encoded_imgs, y_test, '0')
imageGenerator(n)
imageGeneratorTailored()
