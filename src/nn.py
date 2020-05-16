import numpy as np

def gen_layer(min_neurons, max_neurons):
    num_neurons = np.random.randint(min_neurons, max_neurons + 1)
    weights = np.random.uniform(-1, 1, num_neurons)
    return weights

def sigmoid(x):
    pass

# For now, each hidden layer will 
class NeuralNetwork:
    def __init__(self, num_hidden_layers, min_neurons, max_neurons):
        self.num_hidden_layers = num_hidden_layers
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons

        self.weights = []
        for layer in range(num_hidden_layers):
            self.weights.append(gen_layer(min_neurons, max_neurons))

    def print_weights(self):
        for layer in self.weights:
            print(layer)


# Driver code, TEMP
nn = NeuralNetwork(2, 3, 4)
nn.print_weights()
