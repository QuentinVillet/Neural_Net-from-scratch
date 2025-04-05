import numpy as np
layer_outputs = [[4.8, 1.21, 2.385], [8.9, 1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

# we want to get a diff value for each output

print(np.sum(layer_outputs, axis=1, keepdims=True)) # sum of each row
# takes sum of each row and keeps the dimensions of the array
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values) #returns the actual normalised values
# explosion of values w exponentiation, which is an issue and leads to an overflow
# so we can take all vlaues of input layer and subtract the largest value to all of them and thus the largest value will be 0 and so the range of values will only be between 0 and 1

