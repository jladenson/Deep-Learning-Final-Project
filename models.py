import tensorflow as tf
import keras
from keras import layers

class Conv(keras.Model):
    def __init__(self, embedding_dim=4, filter_len=4):
        super(Conv, self).__init__()
        self.conv = keras.Sequential([
            layers.Conv2D(filters=32,
                          kernel_size=(embedding_dim, filter_len),
                          strides=1, # ?
                          activation='relu'),
            layers.MaxPool2D(pool_size=(1, 4)),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(2)
        ])

    def call(self, x):
        return self.conv(x)

class MLP(keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = keras.Sequential([
            layers.Dense(2, activation='relu'),
            layers.Dense(2) # changed to softmax instead of single neuron
        ])

    def call(self, x):
        return self.mlp(x)