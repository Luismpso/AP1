import numpy as np
from layers import Layer

class ActivationLayer(Layer):
    def __init__(self, activation_function, derivative):
        super().__init__()
        self.activation_function = activation_function
        self.derivative = derivative

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation_function(self.input)
        return self.output

    # AQUI ESTAVA O ERRO! Removemos o learning_rate desta linha:
    def backward_propagation(self, output_error): 
        return self.derivative(self.input) * output_error

# Funções auxiliares matemáticas
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x >= 0, 1, 0)

def softmax(x):
    # Usamos o max para estabilidade numérica
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Classes prontas a usar na rede
class ReLUActivation(ActivationLayer):
    def __init__(self):
        super().__init__(relu, relu_derivative)

class SoftmaxActivation(ActivationLayer):
    def __init__(self):
        # A derivada do Softmax em combinação com a CrossEntropy é 1,
        # pois a matemática cancela-se de forma elegante!
        super().__init__(softmax, lambda x: 1)