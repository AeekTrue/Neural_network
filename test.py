from neuromodule import Network
import numpy as np
import load

LEARNING_RATE = float(input("Enter learning rate: "))
MINI_BATCH_SIZE = int(input("Enter mini batch size: "))
MAX_EPOCHS = int(input("Enter max epochs: "))

TRAINING_DATA = load.TRAINING_DATA # load_lesson('')
TEST_DATA = load.TEST_DATA # load_lesson('')

net = Network.Network(np.array([784, 16, 16, 10]))
print('start')

net.SGD(TRAINING_DATA, MINI_BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, test_data=TEST_DATA)

input('Fifish...')

