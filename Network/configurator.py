import numpy as np
from time import strftime
import os
from .logger import configurator_logger


def save_neural_network(network):
    directory = strftime("%Y%m%d_%H%M%S_neural_network")
    os.mkdir(directory)
    path = os.path.join('.', directory)
    weights, biases = network.weights, network.biases
    np.save(os.path.join(path, 'weights.npy'), weights)
    np.save(os.path.join(path, 'biases.npy'), biases)
    configurator_logger.info(f'Neural network was saved in {path}')


def load_neural_network(path):
    weights_path = os.path.join(path, 'weights.npy')
    biases_path = os.path.join(path, 'biases.npy')
    weights = np.load(weights_path, allow_pickle=True)
    biases = np.load(biases_path, allow_pickle=True)
    configurator_logger.info(f'Neural network was loaded from {path}')
    return weights, biases
