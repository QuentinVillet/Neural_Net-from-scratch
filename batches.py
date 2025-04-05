import numpy as np
# many readings of censors at once in a batch. inputs are features from a single sample. Batch just sends multiple samples at once
# putting all the inputs at once rather than in batches will lead to overfitting
# batch size of 32 is common
inputs = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]] #batch of inputs

weights = [[0.2, 0.8, -0.5, 1.0],[0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]
          ] #matrix form


# we need to transpose the weights matrix to match the dimensions of the inputs matrix
# the weights matrix is 3x4 and the inputs matrix is 3x4

biases = [2, 3, 0.5] #vector form

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# matrix product: take first row vector of matrix A and do dot product with the column vector of B and it becomes a scalar of the first element of the output matrix in the dot product matrix
# sizes need to match at the with the second dimension of the first matrix and the first dimension of the second matrix

# to add another layer we need to add a new set of weights and biases
# the output of the first layer becomes the input of the second layer

weights2 = [[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)
