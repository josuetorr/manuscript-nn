import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return x * (1 - x)

""" A simple neural network, it will have 2 layers. The first layer will have 10 neurons and 
the second layer will have 5. These numbers are arbitrary"""
class NeuralNetwork():
    def __init__(self, input_size, output_size):
        # hyper parameters
        self.learning_factor = 0.2
        self.size_i = input_size
        self.size_o = output_size
        self.num_n1 = 10
        self.num_n2 = 10

        self.w1 = np.random.uniform(-1,1,size=(self.size_i, self.num_n1))
        self.w2 = np.random.uniform(-1,1,size=(self.num_n1, self.num_n2))
        self.w3 = np.random.uniform(-1,1,size=(self.num_n2, self.size_o))

        self.hidden_layer1 = [0 for i in range(self.num_n1)]
        self.hidden_layer2 = [0 for i in range(self.num_n2)]
        self.output_layer = [0 for i in range(self.size_o)]

        self.delta1 = [0 for i in range(self.num_n1)]
        self.delta2 = [0 for i in range(self.num_n2)]
        self.delta3 = [0 for i in range(self.size_o)]

    """ Forward propagation """
    def think(self, ex_in):
        self.hidden_layer1 = sigmoid( np.dot( ex_in, self.w1 ) )
        self.hidden_layer2 = sigmoid( np.dot( self.hidden_layer1, self.w2 ) )
        self.output_layer = sigmoid( np.dot( self.hidden_layer2, self.w3 ) )
        return self.output_layer

    def back_prop(self, ex_out):
        self.delta3 = sigmoid_prime( self.output_layer ) \
                * (ex_out - self.output_layer)
        
        self.delta2 = sigmoid_prime( self.hidden_layer2 ) * np.dot( self.w3, self.delta3 )
        self.delta1 = sigmoid_prime( self.hidden_layer1 ) * np.dot( self.w2, self.delta2 )
        return

    def update_weights(self):
        self.w1 = self.w1 + self.learning_factor * np.dot( self.hidden_layer1, np.transpose(self.delta1) )
        self.w2 = self.w2 + self.learning_factor * np.dot( self.hidden_layer2, np.transpose(self.delta2) )
        self.w3 = self.w3 + self.learning_factor * np.dot( self.output_layer, np.transpose(self.delta3) )
        return

    def train(self, ex_in, ex_out):
        self.think(ex_in)
        self.back_prop(ex_out)
        self.update_weights()
        return
