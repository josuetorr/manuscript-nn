import numpy as np
from nn2 import NeuralNetwork
from mnist import MNIST

""" Convert the image to black and white (0 = white, 1 = black)"""
def format_input(image):
    for i in range(len(image)):
        if image[i] != 0:
            image[i] = 1
    return image

""" make a single number into and array of 10 digits, the actual label will
set the index in the list to 1 """
def format_label(label):
    f_label = [0 for i in range(10)]
    f_label[label - 1] = 1
    return f_label

def display_image(image):
    image_dimension = 28
    for i in range(image_dimension):
        for j in range(image_dimension):
            if image[i*image_dimension+j] != 0:
                print('@', end=' ')
            else:
                print('.', end = ' ')
        print()

mndata = MNIST('./data')
images, labels = mndata.load_training()

input_size = len(images[0])
output_size = 10

nn = NeuralNetwork(input_size, output_size)
# let's train for 100 exemples
for i in range(40000):
    formatted_input = format_input(images[i])
    formatted_label = format_label(labels[i])

    nn.train(formatted_input, formatted_label)

# let's use the next 100 exemple as testing data
min_bound = 40000
for i in range(min_bound, min_bound + 50):
    print('Test {}: '.format(i - min_bound + 1))
    print('Predicted value:', np.argmax(nn.think(format_input(np.array(images[i])))))
    print('Actual value:', labels[i])
    print('----------')
