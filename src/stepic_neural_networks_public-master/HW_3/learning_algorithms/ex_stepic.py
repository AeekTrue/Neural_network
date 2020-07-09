import numpy as np
from network import *


w = [np.array([
        [2, 2, 2],
        [2, 2, 2]
    ]),
    np.array([[1, 1]])]

b = [np.zeros(2),
     np.zeros(1)]

d = [(np.array([0, 1, 2]), 1)]

n = Network([3, 2, 1])
# n.weights = w
# n.biases = b
nabla_w, nabla_b = n.backprop(np.array([[0], [1], [2]]), 1)
print(n.biases)
# print(nabla_w)
