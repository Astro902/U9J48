import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.datasets import mnist


(x_train, _), (x_test, _) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data

class Autoencoder(tf.keras.Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    # Encoder
    self.flatten = Flatten()
    self.encoder1 = Dense(128, activation='relu')
    self.encoder2 = Dense(32, activation='relu')

    # Decoder
    self.decoder1 = Dense(128, activation='relu') # First decoder layer
    self.decoder2 = Dense(784, activation='sigmoid') # Second decoder layer
    self.reshape = Reshape((28, 28))

  def call(self, inputs):
    x = self.flatten(inputs)
    x = self.encoder1(x)
    encoded = self.encoder2(x)
    x = self.decoder1(encoded)
    decoded = self.decoder2(x)
    return self.reshape(decoded)
  
  def encode(self, inputs):
    x = self.flatten(inputs)
    x = self.encoder1(x)
    encoded = self.encoder2(x)
    return encoded
  
  def decode(self, encoded):
    x = self.decoder1(encoded)
    decoded = self.decoder2(x)
    return self.reshape(decoded)

#/ Create an instance of the autoencoder
autoencoder = Autoencoder()

#/ Compile the autoencoder
autoencoder.compile(
  optimizer='adam', loss='binary_crossentropy'
)

# Train the autoencoder
autoencoder.fit(
  x_train, x_train,  # Note that both inputs and targets are x_train
  epochs=5,
  shuffle=True,
  validation_data=(x_test, x_test)
)

# Now the autoencoder can be used to encode and decode MNIST images
x_flattened = autoencoder.flatten(x_test)
x_encoded1 = autoencoder.encoder1(x_flattened)
encoded_imgs = autoencoder.encoder2(x_encoded1).numpy()

x_decoded1 = autoencoder.decoder1(encoded_imgs)
decoded_imgs = autoencoder.decoder2(x_decoded1)
decoded_imgs = autoencoder.reshape(decoded_imgs).numpy()

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

def imageGenerator(n):
  latent_dim = 32  # Dimension of the latent space
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

imageDisplay(n)
imageGenerator(n)
