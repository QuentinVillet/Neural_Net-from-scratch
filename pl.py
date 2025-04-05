import sys
import numpy as np
import matplotlib.pyplot as plt

# each input has a unique weight associated with it
# unique inputs (outputs from 3 neurons in previous layer)
inputs = [1, 2, 3, 2.5]
# unique weights for each input
weights = [0.2, 0.8, -0.5, 1.0]
# each neuron has a unique bias
bias = 2

# every input has its own unique weight, but every neuron has a unique bias
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2]
inputs[3]*weights[3] + bias
print(output)
