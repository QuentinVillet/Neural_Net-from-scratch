import numpy as np
softmax_outputs = np.array([0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08])
class_targets = [0,1,1]
# We can use the softmax outputs and the class targets to calculate the negative log likelihood
# This is the same as categorical cross entropy
# The negative log likelihood is calculated as follows:
# -log(p(y|x))
# where p(y|x) is the probability of the target class given the input
neg_log_likelihood = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])

# the problem here is that the log(0) is undefined
# so we need to add a small value to the softmax outputs to avoid this
# this is called epsilon
# we can clip with 1e-7 for example from 0
# to get the accuracy we can use the argmax function to get the index of the maximum value in the softmax outputs
# this will give us the predicted class
predicted_class = np.argmax(softmax_outputs, axis=1)
accuracy = np.mean(predicted_class == class_targets)
# loss metric is more important than the accuracy metric in Neural Network, and to finetune it we can adjust the weights and biases with optimisation
