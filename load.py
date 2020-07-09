from mnist import MNIST
import numpy as np
# import matplotlib.pyplot as plt
num_examples = 16000
num_tests = 2000
mndata = MNIST(".\\src\\traininig_data", gz=True)
images, labels = mndata.load_training()
numbers = np.array(images[:num_examples + num_tests])
answers = np.array(labels[:num_examples + num_tests])

numbers = numbers.reshape((num_examples + num_tests, 784))
answers = answers.reshape((num_examples + num_tests, 1))

numbers = numbers > 100
numbers = numbers.round()

DATA = np.concatenate([numbers, answers], axis=1)

TRAINING_DATA = DATA[:num_examples]
TEST_DATA = DATA[num_examples: num_examples + num_tests]
