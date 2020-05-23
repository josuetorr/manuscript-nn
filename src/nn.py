import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return x * (1 - x)

class NeuralNetwork():
    def __init__(self, size_input=1, size_output=1, num_hidden_layers=1, neurons_per_layers=1):
        self.size_i = size_input
        self.size_o = size_output
        self.num_hl = num_hidden_layers
        self.npl = neurons_per_layers

        # we now know the "dimensions" of our nn
        # let's initialize the activation values of our layers to 0 given the current dimensions
        self._init_layers()

        # let's do the same for the weights, except that we initialize our weight values to a random number
        # between -1 and 1
        self._init_weights()

    def _init_weights(self):
        self.weights = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                shape = (self.size_i, len(self.layers[i+1]))
                self.weights.append(np.random.uniform(-1, 1, (shape)))
            elif i < len(self.layers) - 1:
                shape = (len(self.layers[i]), len(self.layers[i+1]))
                self.weights.append(np.random.uniform(-1, 1, (shape)))


    def _init_layers(self):
        self.layers = []
        for i in range(self.num_hl + 2):
            layer = []
            if i == 0:
                layer = [0] * self.size_i
            elif i == self.num_hl + 1:
                layer = [0] * self.size_o
            else:
                layer = [0] * self.npl
            self.layers.append(layer)

    def forward_prop(self, ex_in, ex_out):
        for i, layer in enumerate(self.layers):
            # setting first layer (input)
            if i == 0:
                self.layers[0] = ex_in
            else:
                for node in layer:
                    activations = sigmoid(np.dot(np.transpose(self.layers[i-1]), self.weights[i-1]))
                    self.layers[i] = activations

    def back_prop(self, ex_in, ex_out):


# drive code, TEMP
if __name__ == '__main__':
    ex_in = [1,0,0]
    ex_out = [1,1,1]
    nn = NeuralNetwork(3,3,1,2)
    print(nn.layers)
    nn.forward_prop(ex_in, ex_out)
    print(nn.layers)
