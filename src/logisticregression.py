import numpy as np


class LogisticRegression:
    """
    Regressão Logística Multi-Classe (Softmax Regression) em NumPy puro.

    Baseline exigido pelo enunciado (Tarefa 2).
    Generaliza a regressão logística binária para K classes usando a função softmax
    e a loss Categorical Cross-Entropy.

    Parâmetros
    ----------
    n_features : int
        Número de features de entrada.
    n_classes : int
        Número de classes (ex: 5 para Human/Google/Meta/OpenAI/Anthropic).
    learning_rate : float
        Taxa de aprendizagem para o gradiente descendente.
    l2_lambda : float
        Parâmetro de regularização L2 (0.0 = sem regularização).
    """

    def __init__(self, n_features, n_classes, learning_rate=0.01, l2_lambda=0.0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda

        # Inicialização dos pesos (pequenos valores aleatórios)
        limit = np.sqrt(2.0 / n_features)
        self.weights = np.random.randn(n_features, n_classes) * limit
        self.biases = np.zeros((1, n_classes))

        self.history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    # ─── Softmax ──────────────────────────────────────────────────────

    def _softmax(self, z):
        """Softmax com estabilidade numérica."""
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    # ─── Cross-Entropy Loss ───────────────────────────────────────────

    def _compute_loss(self, Y_true, Y_pred):
        """
        Categorical Cross-Entropy:
            J = -(1/m) * Σ Σ y_ij * log(ŷ_ij)

        Com regularização L2:
            J += (λ/2m) * Σ w²
        """
        m = Y_true.shape[0]
        epsilon = 1e-15
        Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)

        loss = -np.sum(Y_true * np.log(Y_pred)) / m

        if self.l2_lambda > 0:
            loss += (self.l2_lambda / (2 * m)) * np.sum(self.weights ** 2)

        return loss

    # ─── Acurácia ─────────────────────────────────────────────────────

    def _compute_accuracy(self, Y_true, Y_pred):
        predictions = np.argmax(Y_pred, axis=1)
        labels = np.argmax(Y_true, axis=1)
        return np.mean(predictions == labels)

    # ─── Treino ───────────────────────────────────────────────────────

    def fit(self, X, Y, epochs=100, batch_size=64, verbose=True,
            X_val=None, Y_val=None, patience=None):
        """
        Treina por mini-batch gradient descent.

        Gradiente (Softmax + Cross-Entropy):
            dW = (1/m) * Xᵀ(Ŷ - Y) + λ/m * W
            db = (1/m) * Σ(Ŷ - Y)

        Parâmetros
        ----------
        X : np.ndarray (m, n_features)
        Y : np.ndarray (m, n_classes) — one-hot encoded
        epochs : int
        batch_size : int
        X_val, Y_val : dados de validação (opcionais)
        patience : int ou None — early stopping (None = desligado)
        """
        best_val_loss = np.inf
        patience_counter = 0
        best_weights = None
        best_biases = None
        use_es = patience is not None and X_val is not None

        for epoch in range(epochs):
            # Baralhar dados
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            # Mini-batch training
            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                m_batch = X_batch.shape[0]

                # Forward
                z = X_batch.dot(self.weights) + self.biases
                Y_pred = self._softmax(z)

                # Gradientes
                error = Y_pred - Y_batch
                grad_w = X_batch.T.dot(error) / m_batch
                grad_b = np.sum(error, axis=0, keepdims=True) / m_batch

                # Regularização L2 (não regularizar os biases)
                if self.l2_lambda > 0:
                    grad_w += (self.l2_lambda / m_batch) * self.weights

                # Atualizar pesos
                self.weights -= self.learning_rate * grad_w
                self.biases -= self.learning_rate * grad_b

            # Métricas no fim da época
            Y_pred_full = self.predict_proba(X)
            train_loss = self._compute_loss(Y, Y_pred_full)
            train_acc = self._compute_accuracy(Y, Y_pred_full)
            self.history['loss'].append(train_loss)
            self.history['acc'].append(train_acc)

            if X_val is not None and Y_val is not None:
                Y_pred_val = self.predict_proba(X_val)
                val_loss = self._compute_loss(Y_val, Y_pred_val)
                val_acc = self._compute_accuracy(Y_val, Y_pred_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Early stopping
                if use_es:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_weights = self.weights.copy()
                        best_biases = self.biases.copy()
                    else:
                        patience_counter += 1

                if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch+1:4d}/{epochs} | "
                          f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
            else:
                if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch+1:4d}/{epochs} | "
                          f"Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")

            if use_es and patience_counter >= patience:
                if verbose:
                    print(f"\nEarly Stopping na epoca {epoch+1}! "
                          f"Melhor Val Loss: {best_val_loss:.4f}")
                break

        # Restaurar melhores pesos
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases
            if verbose:
                print("Melhores pesos restaurados.")

    # ─── Previsão ─────────────────────────────────────────────────────

    def predict_proba(self, X):
        """Retorna probabilidades (softmax) para cada classe."""
        z = X.dot(self.weights) + self.biases
        return self._softmax(z)

    def predict(self, X):
        """Retorna os índices das classes previstas."""
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X, Y):
        """Retorna (loss, accuracy) para um conjunto de dados."""
        Y_pred = self.predict_proba(X)
        loss = self._compute_loss(Y, Y_pred)
        acc = self._compute_accuracy(Y, Y_pred)
        return loss, acc