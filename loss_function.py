# We use categorical cross entropy to get the loss for NN as it is very useful
# Li = -log(y^i,k)
# OHE -> n-classes long if you have 3 classes you have a vector 3 values long
# log without base number is ln(x) log based e

'''log solves for x:  e^x = b'''
import numpy as np

b =5.2
print(np.log(b))

# OHE = [1, 0, 0]
# Pred: [0.7, 0.1, 0.2]
# categorical cross entropy = -(1*log(0.7) +0*log(0.1) + 0*log(0.2))
# =0.35667
