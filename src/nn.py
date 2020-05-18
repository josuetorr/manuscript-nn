import numpy as np

# Generate a random number of weights between min_neurons and max_neurons
# Each weight has a value between -1 and 1
def gen_layer(min_neurons, max_neurons):
    num_neurons = np.random.randint(min_neurons, max_neurons + 1)
    layer = np.zeros((num_neurons, ))
    return layer

def gen_weight_layer(num_rows, num_cols):
    if num_rows >= num_cols:
        return np.random.uniform(-1, 1, (num_rows, num_cols))
    else:
        return np.random.uniform(-1, 1, (num_cols, num_rows))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    pass

# For now, each hidden layer will have a random number of neurons between min_neurons and max_neurons, inclusively
class NeuralNetwork:
    def __init__(self, num_hidden_layers=1, min_neurons=1, max_neurons=1, input_layer=None, output_layer=None):

        if input_layer is not None:
            self.set_input_layer(input_layer)
        if output_layer is not None:
            self.set_output_layer(output_layer)

        self.num_hidden_layers = num_hidden_layers
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons

        self.layers = []
        for layer in range(num_hidden_layers):
            self.layers.append(gen_layer(min_neurons, max_neurons))

        self.layers = np.array(self.layers)

    def set_input_layer(self, input_layer):
        self.input_layer = input_layer

    def set_output_layer(self, output_layer):
        self.output_layer = output_layer

    def init_weights(self):
        self.weights = []

        self.weights.append(gen_weight_layer(len(self.input_layer), len(self.layers[0])))

        for i in range(len(self.layers)):
            # will revise this method. It is suboptimal, but will do for now
            weight_layer = []
            if i < len(self.layers) - 1:
                weight_layer = gen_weight_layer(len(self.layers[i]), len(self.layers[i + 1]))
                self.weights.append(weight_layer)

        self.weights.append(gen_weight_layer(len(self.layers[len(self.layers) - 1]), len(self.output_layer)))

    def forward_prop(self):
        # weights have already been randomly initilized
        # input layer has already been setup

        # for each layer, hidden and output, we start the foward propagation
        pass

    def print_layers(self):
        print(self.layers)

    def print_weights(self):
        print(self.weights)



# Driver code, TEMP
if __name__ == '__main__':
    nn = NeuralNetwork(1, 2, 2,[0,0,0], [0,0,0])
    nn.print_layers()
    nn.init_weights()
    nn.print_weights()
