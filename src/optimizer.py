import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Inicializa o otimizador SGD com Momentum.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.retained_gradient = None # Acumula o gradiente de épocas anteriores

    def update(self, weights, gradient):
        """
        Atualiza o gradiente retido, calcula e devolve os pesos atualizados.
        """
        # Inicializa o retained_gradient a zeros na primeira iteração
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros_like(weights)
        
        # Vt = beta * V_{t-1} + (1 - beta) * gradient
        self.retained_gradient = (self.momentum * self.retained_gradient) + ((1 - self.momentum) * gradient)
        
        # Wt = W_{t-1} - alpha * Vt
        updated_weights = weights - (self.learning_rate * self.retained_gradient)
        
        return updated_weights