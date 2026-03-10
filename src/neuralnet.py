import numpy as np


class NeuralNetwork:
    def __init__(self, loss_func):
        self.layers = []
        self.loss_func = loss_func
        self.history = {'loss': [], 'val_loss': []}

    def add(self, layer):
        """Adiciona uma nova camada a rede."""
        self.layers.append(layer)

    def set_training(self, training):
        """Define o modo de treino ou avaliacao para todas as camadas."""
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(training)

    def forward_propagation(self, X):
        """Forward propagation: passa por todas as camadas."""
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output

    def backward_propagation(self, loss_derivative):
        """Backward propagation: propaga o erro de tras para a frente."""
        error = loss_derivative
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)

    def get_mini_batches(self, X, Y, batch_size):
        """Gera os mini-batches baralhando os dados."""
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        for start_idx in range(0, X.shape[0], batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            yield X[batch_idx], Y[batch_idx]

    def _save_weights(self):
        """Guarda uma copia dos pesos de todas as DenseLayers."""
        saved = []
        for layer in self.layers:
            if hasattr(layer, 'get_weights'):
                saved.append(layer.get_weights())
            else:
                saved.append(None)
        return saved

    def _restore_weights(self, saved):
        """Restaura os pesos guardados."""
        for layer, weights in zip(self.layers, saved):
            if weights is not None and hasattr(layer, 'set_weights'):
                layer.set_weights(weights[0], weights[1])

    def fit(self, X, Y, epochs, batch_size=64, verbose=True,
            X_val=None, Y_val=None, patience=None):
        """
        Treina a rede neuronal com suporte a validacao e early stopping.

        patience: numero de epocas sem melhoria na val_loss antes de parar.
                  Se None, nao usa early stopping.
        """
        best_val_loss = np.inf
        patience_counter = 0
        best_weights = None
        use_early_stopping = patience is not None and X_val is not None and Y_val is not None

        for epoch in range(epochs):
            # Modo treino
            self.set_training(True)

            # Mini-batch training
            for X_batch, Y_batch in self.get_mini_batches(X, Y, batch_size):
                y_pred_batch = self.forward_propagation(X_batch)
                loss_derivative = self.loss_func.derivative(Y_batch, y_pred_batch)
                self.backward_propagation(loss_derivative)

            # Modo avaliacao para calcular metricas
            self.set_training(False)

            # Loss de treino
            y_pred_full = self.forward_propagation(X)
            epoch_loss = self.loss_func.loss(Y, y_pred_full)
            self.history['loss'].append(epoch_loss)

            # Loss de validacao
            if X_val is not None and Y_val is not None:
                y_pred_val = self.forward_propagation(X_val)
                val_loss = self.loss_func.loss(Y_val, y_pred_val)
                self.history['val_loss'].append(val_loss)

                # Early Stopping
                if use_early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_weights = self._save_weights()
                    else:
                        patience_counter += 1

                if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                    train_acc = self._compute_accuracy(Y, y_pred_full)
                    val_acc = self._compute_accuracy(Y_val, y_pred_val)
                    print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
            else:
                if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f}")

            # Verificar early stopping
            if use_early_stopping and patience_counter >= patience:
                if verbose:
                    print(f"\nEarly Stopping na epoca {epoch + 1}! Val Loss nao melhorou ha {patience} epocas.")
                    print(f"Melhor Val Loss: {best_val_loss:.4f}")
                break

        # Restaurar os melhores pesos
        if best_weights is not None:
            self._restore_weights(best_weights)
            if verbose:
                print("Melhores pesos restaurados.")

    def _compute_accuracy(self, Y_true, Y_pred):
        predictions = np.argmax(Y_pred, axis=1)
        labels = np.argmax(Y_true, axis=1)
        return np.mean(predictions == labels)

    def evaluate(self, X, Y):
        """Avalia o modelo calculando a perda e a precisao."""
        self.set_training(False)
        y_pred = self.forward_propagation(X)
        loss = self.loss_func.loss(Y, y_pred)

        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(Y, axis=1)
        accuracy = np.mean(predictions == labels)

        return loss, accuracy

    def predict(self, X):
        """Faz previsoes para novos dados."""
        self.set_training(False)
        return self.forward_propagation(X)
