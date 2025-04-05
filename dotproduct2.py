import numpy as np
# dot product of layer

inputs = [1, 2, 3, 2.5] #vector form
weights = [[0.2, 0.8, -0.5, 1.0],[0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]
          ] #matrix form

biases = [2, 3, 0.5] #vector form

output = np.dot(weights, inputs) + biases
print(output)

# output = np.dot(weights, inputs) + biases = [np.dot(weights[0], inputs) + biases[0], np.dot(weights[1], inputs) + biases[1], np.dot(weights[2], inputs) + biases[2]]
# = [2.8, -1.79, 1.885]
# the dot product of the weights and inputs is [2.8, -1.79, 1.885]
# bias helps determine in activation function if the neuron should fire or not

#so far I have only done the dot product of the weights and inputs for a sinlge layer, but in more complex NN we go by batches
# so therefore input data instead of being a vector of 1d array. Instead now, input data is a batch of inputs which is a 2d array or matrix.
