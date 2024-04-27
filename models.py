import tensorflow as tf
import keras
from keras.layers import (Conv2D, Dropout, Concatenate,
                          Flatten, Dense, MaxPool2D)

'''
class Conv(keras.Model):
    def __init__(self, embedding_dim=4, filter_len=4):
        super(Conv, self).__init__()
        self.conv = keras.Sequential([
            Conv2D(filters=32,
                          kernel_size=(embedding_dim, filter_len),
                          strides=1,
                          activation='relu'),
            MaxPool2D(pool_size=(1, 4)),
            Dropout(0.3),
            Flatten(),
            Dense(500),
            Dense(2, activation='softmax')
        ])

    def call(self, x):
        return self.conv(x)
'''

class Conv(tf.keras.Model):
    def __init__(self, embedding_dim=4, filter_lens=[4] * 8):
        super(Conv, self).__init__()

        # Define convolutional layers with max pooling and dropout
        self.conv_pool_layers = []
        for i, filter_len in enumerate(filter_lens):
            conv_layer = Conv2D(32, (embedding_dim, filter_len), activation='relu', name=f'conv2d_{i}')
            pool_layer = MaxPool2D(pool_size=(1, 4), name=f'max_pool2d_{i}')
            dropout_layer = Dropout(0.3, name=f'dropout_{i}')
            self.conv_pool_layers.append([conv_layer, pool_layer, dropout_layer])

        # Define concatenate layer
        self.concatenate = Concatenate(axis=2, name='concatenate')

        # Define flatten layer
        self.flatten_layer = Flatten(name='flatten')

        # Define final dropout layer
        self.dense = Dense(500, activation='relu', name='dense')

        # Define output layer
        self.output_layer = Dense(2, activation='softmax', name='output')

    def call(self, inputs):
        # Forward pass
        conv_pool_outputs = []
        for layers in self.conv_pool_layers:
            outputs = tf.identity(inputs) # copy inputs
            for layer in layers: # does conv, pool, dropout
                outputs = layer(outputs)
            conv_pool_outputs.append(outputs)
        concatenated_output = self.concatenate(conv_pool_outputs)
        flattened_output = self.flatten_layer(concatenated_output)
        dense = self.dense(flattened_output)
        return self.output_layer(dense)

class MLP(keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = keras.Sequential([
            Dense(2, activation='relu'),
            Dense(2, activation='softmax')
        ])

    def call(self, x):
        return self.mlp(x)