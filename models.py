import tensorflow as tf
import numpy as np
import keras
from keras import layers

class Conv(keras.Model):
    def __init__(self, embedding_dim=4, filter_len=4):
        super(Conv, self).__init__()
        self.conv = keras.Sequential([
            layers.Conv2D(32, (embedding_dim, filter_len), activation='relu'),
            layers.MaxPool2D(),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(2, activation='softmax')
        ])
        self.conv.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

    def call(self, x):
        return self.conv(x)
    
class MLP(keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = keras.Sequential([
            layers.Dense(2, activation='relu'),
            layers.Dense(1, activation='relu')
        ])
        self.mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def call(self, x):
        return self.mlp(x)