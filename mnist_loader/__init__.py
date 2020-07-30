from mnist import MNIST
import numpy as np


def get_data():
    """
    Загружаем данные из файлов MNIST.
    """
    mnist_data_training = MNIST("./training_data", gz=True)
    images, labels = mnist_data_training.load_training()
    test_images, test_labels = mnist_data_training.load_testing()

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
    и укажите порог градации (100 по умолчанию).
    """
    # digits = digits > 100
    # digits = digits.round()
    # test_digits = test_digits > 100
    # test_digits = test_digits.round()

    training_data = np.concatenate([digits, answers], axis=1)
    test_data = np.concatenate([test_digits, test_answers], axis=1)
    return training_data, test_data
