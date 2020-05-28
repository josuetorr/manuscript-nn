import numpy as np
from nn import NeuralNetwork as nn
from mnist import MNIST

# Reduce the interval of the mnist data from [0, 255] => [-0.5, 0.5]
def shrink_image_interval(image):
    return (image / 255) - 0.5

# make a single number into and array of 10 digits, the actual label will
# set the index in the list to 1
def format_label(label):
    f_label = [0 for i in range(10)]
    f_label[label - 1] = 1
    return f_label

mndata = MNIST('./data')
images, labels = mndata.load_training()

input_size = len(images[0])

nn = nn(input_size, size_output=10, num_hidden_layers=1, neurons_per_layer=5)
# let's train for 100 exemples
for i in range(100):
    formatted_input = shrink_image_interval(np.array(images[i]))
    formatted_label = format_label(labels[i])

    nn.train(formatted_input, formatted_label)

# let's use the next 100 exemple as testing data
# for i in range(100, 150):
print(nn.think(shrink_image_interval(np.array(images[301]))))
print(labels[301])
