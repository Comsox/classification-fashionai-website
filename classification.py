import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess the data: Before using the pixel values of the images as input to the neural network, 
# they are normalized to be in the range [0, 1] by dividing each pixel value by 255. This can help the neural network learn more effectively.
train_images = train_images / 255.0
test_images = test_images / 255.0
