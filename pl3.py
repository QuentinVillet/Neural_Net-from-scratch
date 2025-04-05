inputs = [1, 2, 3, 2.5] #vector form
weights = [[0.2, 0.8, -0.5, 1.0],[0.5, -0.91, 0.26, -0.5]
 ,[-0.26, -0.27, 0.17, 0.87]] #matrix form

biases = [2, 3, 0.5] #vector form

layer_outputs = [] # output of current layer
# zip combines 2 lists element wise
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 # output of given neuron
    # multiply every input by the weight and add them together
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight # multiply the input by the weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)


# Math Behind the Neural Network:
# shape tells you what the size at each dimension is
# a simple list in numpy is an array in real mathematics it is a vector
# a 2d list is a matrix in numpy
# shape in numpy is dimensions, rows, columns
# dot product multiplies 2 matrices together
# dot product of 2 matrices is the sum of the product of the corresponding elements
# the dot product results in a scalar value of the 2 matrices
