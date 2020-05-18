import numpy as np

# Generate a random number of weights between min_neurons and max_neurons
# Each weight has a value between -1 and 1
def gen_layer(min_neurons, max_neurons):
    num_neurons = np.random.randint(min_neurons, max_neurons + 1)
    layer = np.zeros((num_neurons, ))
    return layer

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    pass

# For now, each hidden layer will have a random number of neurons between min_neurons and max_neurons, inclusively
class NeuralNetwork:
    def __init__(self, num_hidden_layers=1, min_neurons=1, max_neurons=1, input_layer=None, output_layer=None):

        if input_layer is not None:
            set_input_layer(input_layer)
        if output_layer is not None:
            set_output_layer(output_layer)

        self.num_hidden_layers = num_hidden_layers
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons

        self.layers = []
        for layer in range(num_hidden_layers):
            self.layers.append(gen_layer(min_neurons, max_neurons))

        self.layers = np.asmatrix(self.layers)

    def set_input_layer(self, input_layer):
        self.input_layer = input_layer

    def set_output_layer(self, output_layer):
        self.output_layer = ouput_layer

    def init_weights(self):
        self.weights = []
        num_weight_layers = self.num_hidden_layers + 1 

        print("Initializing weights to a random number:")
        for i in range(num_weight_layers):
            # will revise this method. It is suboptimal, but will do for now
            if i == 0:
                print("input to 1st weight layer")
            elif i == num_weight_layers - 1:
                print("last weight layer to output layer")
            else:
                print("layer #", i + 1)



    def forward_prop(self):
        # weights have already been randomly initilized
        # input layer has already been setup

        # for each layer, hidden and output, we start the foward propagation
        pass

    def print_layers(self):
        print(self.layers)



# Driver code, TEMP
if __name__ == '__main__':
    nn = NeuralNetwork(2, 3, 4)
    nn.print_layers()
    nn.init_weights()
