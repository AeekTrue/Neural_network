from mnist import MNIST
import numpy as np
# import matplotlib.pyplot as plt

num_examples = 60000
num_tests = 10000

mndata_training = MNIST(".\\src\\traininig_data", gz=True)

images, labels = mndata_training.load_training()
test_images, test_labels = mndata_training.load_testing()

digits = np.array(images[:num_examples])
answers = np.array(labels[:num_examples])
digits = digits.reshape((num_examples, 784))
answers = answers.reshape((num_examples, 1))
digits = digits > 100
digits = digits.round()

test_digits = np.array(test_images[:num_tests])
test_answers = np.array(test_labels[:num_tests])
test_digits = test_digits.reshape((num_tests, 784))
test_answers = test_answers.reshape((num_tests, 1))
test_digits = test_digits > 100
test_digits = test_digits.round()

TRAINING_DATA = np.concatenate([digits, answers], axis=1)
TEST_DATA = np.concatenate([test_digits, test_answers], axis=1)
print("Loaded!")
