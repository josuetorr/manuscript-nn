import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return x * (1 - x)

class NeuralNetwork():
    def __init__(self, size_input=1, size_output=1, num_hidden_layers=1, neurons_per_layer=1):
        self.size_i = size_input
        self.size_o = size_output
        self.num_hl = num_hidden_layers
        self.npl = neurons_per_layer

        self.learning_factor = 0.2
        # we now know the "dimensions" of our nn
        # let's initialize the activation values of our layers to 0 given the current dimensions
        self._init_layers()

        # let's do the same for the weights, except that we initialize our weight values to a random number
        # between -1 and 1
        self._init_weights()

        # initialize the delta values.
        # we simply copy the layers matrix because they have the same dimensions
        self._init_deltas()

    def _init_weights(self):
        self.weights = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                shape = (self.size_i, len(self.layers[i+1]))
                self.weights.append(np.random.uniform(-1, 1, (shape)))
            elif i < len(self.layers) - 1:
                shape = (len(self.layers[i]), len(self.layers[i+1]))
                self.weights.append(np.random.uniform(-1, 1, (shape)))
        return

    def _init_deltas(self):
        self.deltas = self.layers.copy()
        return

    def _init_layers(self):
        self.layers = []
        # +2 for input and output layer
        for i in range(self.num_hl + 2):
            layer = []
            if i == 0:
                layer = [0] * self.size_i
            elif i == self.num_hl + 1:
                layer = [0] * self.size_o
            else:
                layer = [0] * self.npl
            self.layers.append(layer)
        return

    """ Forward propagation """
    def think(self, ex_in):
        for i, layer in enumerate(self.layers):
            # setting first layer (input)
            if i == 0:
                self.layers[0] = ex_in
            else:
                for node in layer:
                    activations = sigmoid(np.dot(np.transpose(self.layers[i-1]), self.weights[i-1]))
                    self.layers[i] = activations
        return self.layers[-1]

    def back_prop(self, ex_in, ex_out):
        for i in range(len(self.deltas) - 1, -1, -1):
            # setting up the last layer
            if i == len(self.deltas) - 1:
                self.deltas[-1] = sigmoid_prime(np.dot(np.transpose(self.layers[-2]), self.weights[-1])) \
                        * (ex_out - self.layers[-1])

            # all other layers
            elif i < len(self.deltas) - 1 and i >= 0:
                self.deltas[i] = sigmoid_prime(np.dot(self.layers[i+1], np.transpose(self.weights[i]))) \
                        * np.dot(self.deltas[i+1], np.transpose(self.weights[i]))

        return

    def update_weights(self):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] += self.learning_factor * self.layers[i][j] * self.deltas[i][j]
        return

    def train(self, ex_in, ex_out):
        self.think(ex_in)
        self.back_prop(ex_in, ex_out)
        self.update_weights()
        return
