import numpy as np
import copy


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Inicializa o otimizador SGD com Momentum Clássico.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, weights, gradient):
        """
        Atualiza os pesos usando a fórmula standard de Machine Learning.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros_like(weights)

        # V = (momentum * V) + gradiente
        self.retained_gradient = (self.momentum * self.retained_gradient) + gradient

        # W = W - (learning_rate * V)
        updated_weights = weights - (self.learning_rate * self.retained_gradient)

        return updated_weights


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Otimizador Adam (Adaptive Moment Estimation).
        Combina as vantagens do AdaGrad e RMSProp.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 1o momento (media dos gradientes)
        self.v = None  # 2o momento (media dos gradientes ao quadrado)
        self.t = 0     # Contador de passos (para bias correction)

    def update(self, weights, gradient):
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.t += 1

        # Atualizar estimativas dos momentos
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        # Bias correction (corrige o vies para os primeiros passos)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Atualizar pesos
        updated_weights = weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated_weights