import numpy as np

inputs = [1, 2, 3, 2.5] #vector form
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = np.dot(weights, inputs) + bias
print(output)

# 0.2*1 + 0.8*2 + -0.5*3 + 1.0*2.5 + 2 = 4.8
# the dot product of the weights and inputs is 4.8
