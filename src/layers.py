from optimizer import SGD
import numpy as np
import copy


class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.training = True

    def forward_propagation(self, input_data):
        raise NotImplementedError

    def backward_propagation(self, output_error):
        raise NotImplementedError

    def set_training(self, training):
        self.training = training


class DenseLayer(Layer):
    def __init__(self, input_shape, n_units, optimizer=None, learning_rate=0.01, momentum=0.9):
        super().__init__()

        # Inicializacao He (melhor para ReLU)
        limit = np.sqrt(2.0 / input_shape)
        self.weights = np.random.randn(input_shape, n_units) * limit
        self.biases = np.zeros((1, n_units))

        # Se nenhum optimizer for passado, usa SGD por defeito (retrocompativel)
        if optimizer is not None:
            self.weights_optimizer = copy.deepcopy(optimizer)
            self.biases_optimizer = copy.deepcopy(optimizer)
        else:
            self.weights_optimizer = SGD(learning_rate, momentum)
            self.biases_optimizer = SGD(learning_rate, momentum)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        input_error = np.dot(output_error, self.weights.T)

        weights_gradient = np.dot(self.input.T, output_error)
        biases_gradient = np.sum(output_error, axis=0, keepdims=True)

        self.weights = self.weights_optimizer.update(self.weights, weights_gradient)
        self.biases = self.biases_optimizer.update(self.biases, biases_gradient)

        return input_error

    def get_weights(self):
        return self.weights.copy(), self.biases.copy()

    def set_weights(self, weights, biases):
        self.weights = weights.copy()
        self.biases = biases.copy()


class DropoutLayer(Layer):
    def __init__(self, rate=0.3):
        """
        Dropout: desliga neuronios aleatoriamente durante o treino.
        rate: fracao de neuronios a desligar (ex: 0.3 = 30%)
        """
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward_propagation(self, input_data):
        self.input = input_data
        if self.training:
            # Criar mascara binomial: 1 = manter, 0 = desligar
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape)
            # Inverted dropout: escalar pelo (1 - rate) para manter a escala
            self.output = (input_data * self.mask) / (1 - self.rate)
        else:
            # Em modo de avaliacao, nao aplicar dropout
            self.output = input_data
        return self.output

    def backward_propagation(self, output_error):
        if self.training:
            return (output_error * self.mask) / (1 - self.rate)
        return output_error
