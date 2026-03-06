from optimizer import SGD
import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_data):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

class DenseLayer(Layer):
    def __init__(self, input_shape, n_units, learning_rate=0.01, momentum=0.9):
        super().__init__()
        self.weights = np.random.randn(input_shape, n_units) * 0.01
        self.biases = np.zeros((1, n_units))
        
        # Instanciamos um otimizador para os pesos e outro para os biases
        self.weights_optimizer = SGD(learning_rate, momentum)
        self.biases_optimizer = SGD(learning_rate, momentum)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        # Derivada em relação aos inputs
        input_error = np.dot(output_error, self.weights.T)
        
        # Derivada em relação aos pesos
        weights_gradient = np.dot(self.input.T, output_error)
        biases_gradient = np.sum(output_error, axis=0, keepdims=True)
        
        # Aqui usamos o nosso novo otimizador!
        self.weights = self.weights_optimizer.update(self.weights, weights_gradient)
        self.biases = self.biases_optimizer.update(self.biases, biases_gradient)
        
        return input_error