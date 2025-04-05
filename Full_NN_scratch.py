import numpy as np
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs  # Store for backpropagation
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues, learning_rate):
        # Gradients
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)  # Pass gradients to previous layer

        # Update parameters
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0  # Gradient is 0 where input is negative

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples

class Loss_CategoricalCrossentropy:
    def calculate(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        loss = -np.log(correct_confidences)
        return np.mean(loss)

def train(X, y, epochs=1000, batch_size=32, learning_rate=0.1):
    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()
    loss_function = Loss_CategoricalCrossentropy()

    for epoch in range(epochs):
        for batch_start in range(0, len(X), batch_size):
            batch_X = X[batch_start:batch_start + batch_size]
            batch_y = y[batch_start:batch_start + batch_size]

            # Forward pass
            dense1.forward(batch_X)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            activation2.forward(dense2.output)

            # Compute loss
            loss = loss_function.calculate(activation2.output, batch_y)

            # Backpropagation
            activation2.backward(activation2.output, batch_y)
            dense2.backward(activation2.dinputs, learning_rate)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    # Predictions: class with highest probability
    predictions = np.argmax(activation2.output, axis=1)
    # Accuracy calculation
    accuracy = np.mean(predictions == y)
    print(f"Final Training Accuracy: {accuracy * 100:.2f}%")

# Generate dataset and train the network
X, y = spiral_data(samples=100, classes=3)
train(X, y, epochs=1000, batch_size=32, learning_rate=0.1)
