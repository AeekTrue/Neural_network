from neuromodule import Network
import numpy as np
import load

print("import successful")

LEARNING_RATE = 1
MINI_BATCH_SIZE = 20
MAX_EPOCHS = 200

TRAINING_DATA = load.TRAINING_DATA # load_lesson('')
TEST_DATA = load.TEST_DATA # load_lesson('')

net = Network.Network(np.array([784, 300, 100, 10]))
print('start')

net.SGD(TRAINING_DATA, MINI_BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, test_data=TEST_DATA)

input('Fifish...')

