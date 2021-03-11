from mnist import MNIST
import numpy as np
from .logger import logger
import os


def get_data():
    """
    Загружает данные из файлов MNIST.
    """
    num_examples = 60000
    num_tests = 10000
    path = os.path.join(__path__[0], "training_data")
    mnist_data_training = MNIST(path, gz=True)
    images, labels = mnist_data_training.load_training()
    test_images, test_labels = mnist_data_training.load_testing()
    logger.success('MNIST is loaded')
    """
    Создаем из загруженных данных numpy массивы
    и задаём им формы.
    """
    digits = np.array(images)
    answers = np.array(labels)
    digits = digits.reshape((num_examples, 784))
    answers = answers.reshape((num_examples, 1))

    test_digits = np.array(test_images[:num_tests])
    test_answers = np.array(test_labels[:num_tests])
    test_digits = test_digits.reshape((num_tests, 784))
    test_answers = test_answers.reshape((num_tests, 1))

    """
    Нормализуем данные в диапазоне [0;1].
    """
    digits = digits / 255
    test_digits = test_digits / 255

    """
    Раскомментируйте если нужны двухцветные изображения
    и укажите порог градации (0.5 по умолчанию).
    """
    digits = digits > 0.2
    digits = digits.round()
    test_digits = test_digits > 0.2
    test_digits = test_digits.round()

    training_data = np.concatenate([digits, answers], axis=1)
    test_data = np.concatenate([test_digits, test_answers], axis=1)
    logger.success('Data is created')
    return training_data, test_data
