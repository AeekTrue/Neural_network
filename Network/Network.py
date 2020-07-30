import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Network:
    def __init__(self, structure: np.ndarray, weights=None, biases=None,
                 activate_function=sigmoid,
                 activate_derivative=sigmoid_derivative):
        num_layers = len(structure)

        self.num_layers = num_layers
        self.structure = structure
        self.activate_function = activate_function
        self.activate_derivative = activate_derivative

        if weights is not None:
            self.weights = weights
        else:
            self.weights = [np.random.randn(structure[layer], structure[layer - 1]) for layer in
                            range(1, num_layers)]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [np.random.randn(structure[layer]) for layer in range(1, num_layers)]

    def forward_feed(self, example):
        output = example
        for w, b in zip(self.weights, self.biases):
            output = self.activate_function(np.dot(w, output) + b)
        return output

    def SGD(self, data, mini_batch_size, max_epochs, learning_rate, test_data=None):
        """
        Обучение нейросети методом стохастического градиентного спуска.
        """

        num_examples = np.size(data, axis=0)
        for epoch in range(max_epochs):
            np.random.shuffle(data)
            mini_batches = [data[k: k + mini_batch_size] for k in range(0, num_examples, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data is not None:
                present_success_tests = self.evaluate(test_data)
                print('Epoch {}: {}%'.format(epoch, present_success_tests))
            else:
                print("Epoch {} completed".format(epoch))

        if test_data is not None:
            return self.evaluate(test_data)
        return 0

    def update_mini_batch(self, mini_batch, learning_rate):
        """
        Корректирует параметры нейросети согласно
        производным целевой функции по этим параметрам
        (на len(mini_batch) примерах)

        mini_batch - список примеров и ответов к ним
        learning_rate - константа скорости обучения
        """
        nabla_biases = [np.zeros(biases_layer.shape) for biases_layer in self.biases]
        nabla_weights = [np.zeros(weights_of_layer.shape) for weights_of_layer in self.weights]

        """
        x - текущий пример из mini_batch
        y - ответ к примеру x
        
        Важно понимать: производная ц.ф. по параметру на нескольких примерах - 
        это есть среднее арифметическое производной по этому параметру на каждых примерах в отдельности.
        
        В этом цикле вычисляются матрицы производных ц.ф. на каждом примере и суммируются.
        """
        for x, y in zip(mini_batch[:, :-1], mini_batch[:, -1]):
            nabla_w, nabla_b = self.backprop(x, y)
            nabla_biases = [nbs + nb for nbs, nb in zip(nabla_biases, nabla_b)]
            nabla_weights = [nws + nw for nws, nw in zip(nabla_weights, nabla_w)]

        """
        Усреднение по примерам вынесено в learning_rate дабы не повторять эту операцию в циклах ниже.
        """
        learning_rate = learning_rate / len(mini_batch)

        """
        Все параметры уменьшаются на величину производной по ним (с поправкой на learning_rate). 
        """
        self.weights = [w - learning_rate * nws for w, nws in zip(self.weights, nabla_weights)]
        self.biases = [b - learning_rate * nbs for b, nbs in zip(self.biases, nabla_biases)]

    def backprop(self, x, y):
        """
        Вычистяет производные целевой функции по всем параметрам нейросети
        на одном примере.

        x - пример (вертикальный вектор)
        y - правельный ответ для примера x
        nabla_biases - матрица производных целевой функции по всем смещениям
        nabla_weights - матрица производных целевой функции по всем весам
        """
        nabla_biases = [np.zeros(b.shape) for b in self.biases]
        nabla_weights = [np.zeros(w.shape) for w in self.weights]

        """
        Делаем прямое распространение, сохраняя резыльтаты
        сумматорной и активационной функций каждого нейрона
        в summatory_matrix и activation_matrix соответственно.
        
        Переменные activation и summatory перезаписываются результатами
        работы каждого слоя.
        """
        activation = x
        activation_matrix = [x]
        summatory_matrix = []
        for w, b in zip(self.weights, self.biases):
            summatory = w.dot(activation) + b
            summatory_matrix.append(summatory)
            activation = self.activate_function(summatory)
            activation_matrix.append(activation)

        """ 
        Делаем обратное распространение, сохраняя производные
        целевой функциии (послойно, от последнего к первому)
        в nabla_biases для смещений, в nabla_weights для весов
        """
        delta = self.cost_derivative(activation_matrix[-1], y) * self.activate_derivative(summatory_matrix[-1])
        nabla_biases[-1] = delta
        nabla_weights[-1] = np.outer(delta, activation_matrix[-2])
        for layer in range(2, self.num_layers):
            delta = np.dot(self.weights[-layer + 1].T, delta) * self.activate_derivative(summatory_matrix[-layer])
            nabla_biases[-layer] = delta
            nabla_weights[-layer] = np.outer(delta, activation_matrix[-layer - 1])

        return nabla_weights, nabla_biases

    def evaluate(self, test_data):
        """
        Возвращает процент правильных ответов нейросети на тестовой выборке.
        """
        num_test_examples = np.size(test_data, axis=0)
        test_result = [(np.argmax(self.forward_feed(x)), y) for x, y in zip(test_data[:, :-1], test_data[:, -1])]
        return sum(int(x == y) for (x, y) in test_result) / num_test_examples

    def cost_derivative(self, output_activations, y):
        """
        Возвращает вектор частных производных
        целевой функции по активациям выходного слоя.
        """
        return output_activations - y
