from keras import Model, Sequential, layers

class Conv(Model):
    def __init__(self, embedding_dim=4, filter_lens=[4] * 8):
        super(Conv, self).__init__()

        # Define convolutional layers with max pooling and dropout
        self.conv_pool_layers = []
        for i, filter_len in enumerate(filter_lens):
            conv_layer = layers.Conv2D(32, (embedding_dim, filter_len), activation='relu', name=f'conv2d_{i}')
            pool_layer = layers.MaxPool2D(pool_size=(1, 4), name=f'max_pool2d_{i}')
            dropout_layer = layers.Dropout(0.3, name=f'dropout_{i}')
            self.conv_pool_layers.append([conv_layer, pool_layer, dropout_layer])

        # Define concatenate layer
        self.concatenate = layers.Concatenate(axis=2, name='concatenate')

        # Define flatten layer
        self.flatten_layer = layers.Flatten(name='flatten')

        # Define final dropout layer
        self.dense = layers.Dense(500, activation='relu', name='dense')

        # Define output layer
        self.output_layer = layers.Dense(2, activation='softmax', name='output')

    def call(self, inputs):
        # Forward pass
        conv_pool_outputs = []
        for layer_layers in self.conv_pool_layers:
            outputs = inputs # copy inputs
            for layer in layer_layers: # does conv, pool, dropout
                outputs = layer(outputs)
            conv_pool_outputs.append(outputs)
        concatenated_output = self.concatenate(conv_pool_outputs)
        flattened_output = self.flatten_layer(concatenated_output)
        dense = self.dense(flattened_output)
        return self.output_layer(dense)

class MLP(Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = Sequential([
            layers.Dense(2, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])

    def call(self, x):
        return self.mlp(x)
    
class Simple(Model):
    def __init__(self):
        super(Simple, self).__init__()
        self.simple = Sequential([
            layers.Conv1D(32, 3, activation='relu', name=f'conv1d_1'),
            layers.Dense(2, activation='softmax')
        ])

    def call(self, x):
        return self.simple(x)
