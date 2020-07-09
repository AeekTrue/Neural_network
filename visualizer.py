import matplotlib.pyplot as plt
import numpy as np


def load_lesson(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    header = np.loadtxt(path, delimiter=",", max_rows=1, dtype=str)
    return data, header


def visual_lesson(data, headers):
    examples, answers = data[:, :-1], data[:, -1]
    first_sort = answers == 1
    zero_sort = np.logical_not(first_sort)
    plt.scatter(examples[zero_sort][:, 0], examples[zero_sort][:, 1], label="Zero_sort")
    plt.scatter(examples[first_sort][:, 0], examples[first_sort][:, 1], label="First_sort")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    plt.legend()
    plt.show()


def visual_errors(data, headers, network):
    examples, answers = data[:, :-1], data[:, -1]
    predict = network.forward_feed(examples).round().flatten()
    right = (answers == predict)
    wrong = np.logical_not(right)
    plt.scatter(examples[right][:, 0], examples[right][:, 1], color='blue', label="Right")
    plt.scatter(examples[wrong][:, 0], examples[wrong][:, 1], color='red', label="Wrong")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    file_name = input('Enter file name: ')
    visual_lesson(*load_lesson(file_name))
