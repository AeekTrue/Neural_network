#!/usr/bin/python3
import network
import numpy as np
import mnist_loader


LEARNING_RATE = float(input("Enter learning rate: "))
MINI_BATCH_SIZE = int(input("Enter mini batch size: "))
MAX_EPOCHS = int(input("Enter max epochs: "))

TRAINING_DATA, TEST_DATA = mnist_loader.get_data()

net = network.Network(np.array([784, 16, 16, 10]))

net.SGD(TRAINING_DATA, MINI_BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, test_data=TEST_DATA)
