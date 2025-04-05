import math
import numpy as np
layer_outputs = [4.8, 1.21, 2.385]
layer_outputs = [4.8, 4.79, 4.25] #this is a lot better because the outputs are closer together

# accuracy is not a good metric in NN

# SOFTMAX
# we want to get to a prob. distrubution as the output ideally
# if we usd absolute values or squares for the negative numbers doesnt work in backprop because a -9 is very different from 9
# so we can use y=e^x function so that 0 or negative values are at 0

# first we exponentiate the output values
E = math.e
exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(exp_values)


# exp_values = []
# for output in layer_outputs:
#     exp_values.append(E**output)

# next step is to normalise the values
# exponentiate to get rid of (-) values without losing their values

# norm_base = sum(exp_values)
# norm_values = []

# for value in norm_values:
#     norm_values.append(value / norm_base)

# should be very close to 1 (the sum) and gives us the prob distribution of each outputs

# softmax
# input -> exponentiate e^x .... -> normalize -> output
