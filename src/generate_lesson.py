import numpy as np
import time

num_examples = 1000
num_inputs = 2
prefix = 'circle'  # round(time.time())

training_file_name = f'training_data_{prefix}.csv'
test_file_name = f'test_data_{prefix}.csv'


def sort_func(x, y):
    return (x - 0.5)**2 + (y - 0.5)**2 < 0.1


training = np.random.random((num_examples, num_inputs + 1))
training[:, -1] = sort_func(*training[:, :-1].T)
np.savetxt(training_file_name, training, delimiter=',')

test = np.random.random((num_examples, num_inputs + 1))
test[:, -1] = sort_func(*test[:, :-1].T)
np.savetxt(test_file_name, test, delimiter=',')
