import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Network:
    def __init__(self, structure: np.ndarray, weights: np.ndarray, biases: np.ndarray,
                 activation_function=sigmoid,
                 activation_function_derivative=sigmoid_derivative):
        num_layers = len(structure)

        self.num_layers = num_layers
        self.structure = structure
        self.weights = weights
        self.biases = biases
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def forward_feed(self, example):
        return

