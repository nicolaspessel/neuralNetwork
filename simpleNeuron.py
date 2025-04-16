import numpy as np

def sigmoid(x):
    # Activation function sigmoid: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weight, bias):
        self.weights = weight
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    
weights = np.array([0,1])
bias = 4
n = Neuron(weights, bias)

x = np.array([2,3])
print(n.feedforward(x))