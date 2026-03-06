import numpy as np

class NeuralNetwork:
    def __init__(self, loss_func):
        self.layers = []
        self.loss_func = loss_func
        self.history = {'loss': []} # Dicionário para guardar a loss a cada época 

    def add(self, layer):
        """Adiciona uma nova camada à rede."""
        self.layers.append(layer)

    def forward_propagation(self, X):
        """Forward propagation: passa por todas as camadas."""
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output

    def backward_propagation(self, loss_derivative):
        """Backward propagation: propaga o erro de trás para a frente."""
        # Nota: O learning_rate já não é passado aqui porque o otimizador dentro da camada trata disso!
        error = loss_derivative
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)

    def get_mini_batches(self, X, Y, batch_size):
        """Gera os mini-batches baralhando os dados."""
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices) # Baralhar é crucial para a rede não decorar a ordem
        
        for start_idx in range(0, X.shape[0], batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            yield X[batch_idx], Y[batch_idx]

    def fit(self, X, Y, epochs, batch_size=64, verbose=True):
        """Treina a rede neuronal (Algoritmo completo da Aula 3) [cite: 1268, 1276-1288]."""
        for epoch in range(epochs): # 10. Repetir para todas as épocas
            
            # 1. Computar mini batches
            for X_batch, Y_batch in self.get_mini_batches(X, Y, batch_size):
                
                # 2. Forward propagation para o batch
                y_pred_batch = self.forward_propagation(X_batch)
                
                # 3. Computar a derivada do erro
                loss_derivative = self.loss_func.derivative(Y_batch, y_pred_batch)
                
                # 4. Backpropagate o erro
                self.backward_propagation(loss_derivative)
                
            # 5. Fim do ciclo de batches 
            
            # 6. Computar a loss geral (usando todo o X para monitorizar o treino)
            y_pred_full = self.forward_propagation(X)
            epoch_loss = self.loss_func.loss(Y, y_pred_full)
            
            # 8. Guardar no history
            self.history['loss'].append(epoch_loss)
            
            # 9. Print se verbose for True
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                # (Omissão do passo 7 "métricas" por simplicidade, mas podes adicionar accuracy aqui)
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f}")

    def evaluate(self, X, Y):
        """
        Avalia o modelo calculando a perda e a precisão.
        """
        # 1. Obter previsões
        y_pred = self.forward_propagation(X)
        
        # 2. Calcular a Loss usando a função de perda da rede
        loss = self.loss_func.loss(Y, y_pred)
        
        # 3. Calcular a Accuracy
        # Transformamos One-Hot em índices (ex: [0, 0, 1] vira 2)
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(Y, axis=1)
        accuracy = np.mean(predictions == labels)
        
        return loss, accuracy

    def predict(self, X):
        """Faz previsões para novos dados"""
        return self.forward_propagation(X)