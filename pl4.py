import numpy as np
# pip install nnfs (use nnfs package)
from nnfs.datasets import spiral_data

# nnfs.init() #setting a default data type for numpy to use, and used the for the dataset


# def create_data(points, classes):
#     X = np.zeros((points*classes, 2))
#     y = np.zeros(points*classes, dtype='unit8')
#     for class_number in range(classes):
#         ix = range(points*class_number, points*(class_number+1))
#         r = np.linspace(0.0, 1, points) #radius
#         t = np.linspace(class_number*4, (class_number+1)*4, points)*0.2
#         X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
#         y[ix] = class_number
#     return X, y

# np.random.seed(0)


# input data
# X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]
#      ] #batch of inputs

# inputs = [0,1, -1, -2.7, 1.1, 2.2, -100]
# output = []

# for i in inputs:
#     if i>0:
#         output.append(i)
#     elif i <= 0:
#         output.append(0)
# print(output)


# when we load the model we save the weights and biases
# when making a new neural network we need to initialize the weights and biases
# the weights are initialized with a random number between -1 and 1: if the data is larger, you are making the weights larger. So first good practice is to normalise and scale the data
# biases tend to be initialized with 0: but if your neurons are not firing, you can initialize the biases with a small number, otherwise you will propegate through the network with only 0s.
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
    #    we shaped our weights in a way not to transpose it so we do not need to reinitialize it
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # n inputs, n neurons
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # to prevent overflow
        # [1,0] vs [[0,1][1,0]]
        if len(y_true.shape) == 1: #if y_true is a vector
            correct_confidences = y_pred_clipped[range(samples), y_true] #we want the correct confidence of the predicted class
        elif len(y_true.shape) == 2: # if y_true is a matrix
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1) # we want the sum of the predicted class
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods




X,y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation1= Activation_ReLu()

dense2 = Layer_Dense(3, 3)
activation2= Activation_softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print('Loss:', loss)

# to know how right the probabilities that the NN learned, we need to have a loss function
# input to np.zeroes, the first parameter is the shape vs np.random.randn where the parameters are the shape


# ACTIVATION FUNCTIONS
# step function either 0 or 1
# each neuron in the hidden layers (and output layer) has an activation function, tweaking bias and weights affect the ouptut of the activation function

#sigmoid activation function
# we get more granular output from this function between 0 and 1 (more reliable)

# Rectified Linear function (ReLu)
# if x> 0, ouput is x. If x≤0, ouput is 0
# output can be granular too. Why this over sigmoid? bc sigmoid has vanishing gradient problem.
# ReLu is fast b/c it is simple (less complex than sigmoid)

# why use activation function? to fire a neuron or not
# essentially with linear activation functions we can only fit linear functions.
# if we try to fit a non linear function to a linear activation function, it can't do it (that is why ReLu works great)
# everyhting is built into the fact that all the functions are nonlinear and the rectifiying principle makes it so powerful
# we can offset the activation function by tweaking the weights and at which one it deactivates.
# the bias and weights can offset the function and change the output (transformation) if both weights and bias are 0 or dont change, the output will stay the same in the second layer
# increasing weight makes the slope steeper and vice versa
# to make the activation function bounded, the second neuron can bound it by adjusting the second weight (and the bias)
# bias moves the function up and down
# indiv neurons play a small role in the overall structure of the NN (we can use optimisers to fit the activation functions to match the nonlinear function of the input)
# we need to know the why /debug neural networks to understand how they work and thus creating from scratch allows us to tweak it in case of errors.
# weights and biases are very intersting when they work together even more than within the indiv neurons themselves.
