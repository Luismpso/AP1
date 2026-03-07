import numpy as np

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
        
        # Fórmula standard (sem o 1 - momentum): 
        # V = (momentum * V) + gradiente
        self.retained_gradient = (self.momentum * self.retained_gradient) + gradient
        
        # W = W - (learning_rate * V)
        updated_weights = weights - (self.learning_rate * self.retained_gradient)
        
        return updated_weights