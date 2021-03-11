import numpy as np
from time import strftime
import os
from .logger import logger


def save_neural_network(network, output_dir):
    directory = strftime("%Y%m%d_%H%M%S_neural_network")
    path = os.path.join(output_dir, directory)
    os.mkdir(path)
    weights, biases = network.weights, network.biases
    np.save(os.path.join(path, 'weights.npy'), weights)
    np.save(os.path.join(path, 'biases.npy'), biases)
    logger.info(f'Neural network was saved in {path}')


def load_neural_network(path):
    weights_path = os.path.join(path, 'weights.npy')
    biases_path = os.path.join(path, 'biases.npy')
    weights = np.load(weights_path, allow_pickle=True)
    biases = np.load(biases_path, allow_pickle=True)
    logger.info(f'Neural network was loaded from {path}')
    return weights, biases
