from neuromodule import Network
import numpy as np
from visualizer import visual_errors, load_lesson


LEARNING_RATE = 25
MINI_BATCH_SIZE = 50
MAX_EPOCHS = 200
TRAINING_DATA, HEADERS = load_lesson('src\\data.csv')
TEST_DATA, _ = load_lesson('src\\data.csv')

net = Network.Network(np.array([2, 2, 1]))
net.SGD(TRAINING_DATA, MINI_BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, test_data=TEST_DATA)

visual_errors(TEST_DATA, HEADERS, net)

