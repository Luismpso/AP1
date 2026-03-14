import numpy as np
import re
from collections import Counter


# Vectorizadores de texto em NumPy

class BagOfWords:
    """
    Bag of Words em NumPy puro.

    Conta a frequência de cada palavra no vocabulário para cada documento.
    Equivalente ao CountVectorizer do sklearn.
    """

    def __init__(self, max_features=5000, min_df=2, max_df=0.95, stop_words=None):
        """
        Parâmetros
        ----------
        max_features : int
            Número máximo de palavras no vocabulário.
        min_df : int
            Frequência mínima de documentos para incluir uma palavra.
        max_df : float
            Proporção máxima de documentos (palavras muito comuns são excluídas).
        stop_words : set ou None
            Conjunto de palavras a ignorar. Se None, usa uma lista base de inglês.
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = None  # {palavra: índice}

        if stop_words is None:
            self.stop_words = {
                'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'shall',
                'can', 'this', 'that', 'these', 'those', 'it', 'its', 'i', 'me',
                'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them',
                'their', 'not', 'no', 'nor', 'so', 'if', 'as', 'than', 'then',
                'also', 'just', 'about', 'which', 'what', 'who', 'whom', 'when',
                'where', 'how', 'all', 'each', 'every', 'both', 'more', 'most',
                'other', 'some', 'such', 'only', 'own', 'same', 'too', 'very',
                'up', 'out', 'into', 'over', 'after', 'before', 'between',
                'through', 'during', 'above', 'below', 'any', 'there', 'here',
            }
        else:
            self.stop_words = set(stop_words)

    def _tokenize(self, text):
        """Tokenização simples: minúsculas + apenas alfanuméricos."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def fit(self, texts):
        """
        Constrói o vocabulário a partir dos textos de treino.
        """
        n_docs = len(texts)
        doc_freq = Counter()     # Em quantos documentos aparece cada palavra
        word_freq = Counter()    # Frequência total de cada palavra

        tokenized = []
        for text in texts:
            tokens = self._tokenize(text)
            tokenized.append(tokens)
            unique_tokens = set(tokens)
            doc_freq.update(unique_tokens)
            word_freq.update(tokens)

        # Filtrar por min_df e max_df
        max_doc_count = int(self.max_df * n_docs)
        filtered = {
            word: freq for word, freq in word_freq.items()
            if doc_freq[word] >= self.min_df and doc_freq[word] <= max_doc_count
        }

        # Selecionar as max_features mais frequentes
        top_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        top_words = top_words[:self.max_features]

        self.vocabulary = {word: idx for idx, (word, _) in enumerate(top_words)}
        self._tokenized_cache = tokenized
        return self

    def transform(self, texts):
        """
        Transforma textos numa matriz de contagens (m, vocab_size).
        """
        if self.vocabulary is None:
            raise ValueError("Chamar fit() primeiro.")

        vocab_size = len(self.vocabulary)
        matrix = np.zeros((len(texts), vocab_size), dtype=np.float64)

        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            for token in tokens:
                if token in self.vocabulary:
                    matrix[i, self.vocabulary[token]] += 1

        return matrix

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


class TfidfVectorizer:
    """
    TF-IDF Vectorizer em NumPy puro.

    TF-IDF = Term Frequency × Inverse Document Frequency

    Suporta:
    - Word n-grams e char n-grams (analyzer='word' ou 'char_wb')
    - Sublinear TF: tf = 1 + log(tf) em vez de tf bruto
    - Normalização L2 por documento
    """

    def __init__(self, max_features=5000, min_df=2, max_df=0.95,
                 stop_words=None, ngram_range=(1, 1), sublinear_tf=True,
                 analyzer='word'):
        """
        Parâmetros
        ----------
        max_features : int
            Tamanho máximo do vocabulário.
        min_df : int
            Frequência mínima de documentos.
        max_df : float
            Proporção máxima de documentos.
        stop_words : set ou None
            Stopwords (só usado se analyzer='word').
        ngram_range : tuple (min_n, max_n)
            Ex: (1,2) para unigrams + bigrams.
        sublinear_tf : bool
            Se True, usa 1 + log(tf) em vez de tf bruto.
        analyzer : str
            'word' para word n-grams, 'char_wb' para character n-grams.
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.analyzer = analyzer
        self.vocabulary = None
        self.idf = None

        if stop_words is None and analyzer == 'word':
            self.stop_words = {
                'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'shall',
                'can', 'this', 'that', 'these', 'those', 'it', 'its', 'i', 'me',
                'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them',
                'their', 'not', 'no', 'nor', 'so', 'if', 'as', 'than', 'then',
                'also', 'just', 'about', 'which', 'what', 'who', 'whom', 'when',
                'where', 'how', 'all', 'each', 'every', 'both', 'more', 'most',
                'other', 'some', 'such', 'only', 'own', 'same', 'too', 'very',
                'up', 'out', 'into', 'over', 'after', 'before', 'between',
                'through', 'during', 'above', 'below', 'any', 'there', 'here',
            }
        else:
            self.stop_words = set(stop_words) if stop_words else set()

    def _tokenize_word(self, text):
        """Tokenização de palavras."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        return tokens

    def _generate_word_ngrams(self, tokens):
        """Gera n-grams de palavras."""
        min_n, max_n = self.ngram_range
        ngrams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i:i + n]))
        return ngrams

    def _generate_char_ngrams(self, text):
        """Gera n-grams de caracteres (com word boundaries)."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.split()

        min_n, max_n = self.ngram_range
        ngrams = []
        for word in words:
            # Adicionar espaços como word boundaries (como char_wb do sklearn)
            padded = ' ' + word + ' '
            for n in range(min_n, max_n + 1):
                for i in range(len(padded) - n + 1):
                    ngrams.append(padded[i:i + n])
        return ngrams

    def _extract_ngrams(self, text):
        """Extrai n-grams conforme o analyzer configurado."""
        if self.analyzer == 'word':
            tokens = self._tokenize_word(text)
            return self._generate_word_ngrams(tokens)
        elif self.analyzer == 'char_wb':
            return self._generate_char_ngrams(text)
        else:
            raise ValueError(f"Analyzer '{self.analyzer}' não suportado. Usar 'word' ou 'char_wb'.")

    def fit(self, texts):
        """
        Constrói o vocabulário e calcula o IDF a partir dos textos de treino.

        IDF(t) = log((1 + n) / (1 + df(t))) + 1
        (fórmula smooth, a mesma usada pelo sklearn por defeito)
        """
        n_docs = len(texts)
        doc_freq = Counter()
        total_freq = Counter()

        for text in texts:
            ngrams = self._extract_ngrams(text)
            unique_ngrams = set(ngrams)
            doc_freq.update(unique_ngrams)
            total_freq.update(ngrams)

        # Filtrar por min_df e max_df
        max_doc_count = int(self.max_df * n_docs)
        filtered = {
            ng: freq for ng, freq in total_freq.items()
            if doc_freq[ng] >= self.min_df and doc_freq[ng] <= max_doc_count
        }

        # Top max_features
        top = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        top = top[:self.max_features]

        self.vocabulary = {ng: idx for idx, (ng, _) in enumerate(top)}

        # Calcular IDF (smooth)
        vocab_size = len(self.vocabulary)
        self.idf = np.zeros(vocab_size)
        for ng, idx in self.vocabulary.items():
            df = doc_freq[ng]
            self.idf[idx] = np.log((1 + n_docs) / (1 + df)) + 1

        return self

    def transform(self, texts):
        """
        Transforma textos numa matriz TF-IDF (m, vocab_size).
        """
        if self.vocabulary is None:
            raise ValueError("Chamar fit() primeiro.")

        vocab_size = len(self.vocabulary)
        matrix = np.zeros((len(texts), vocab_size), dtype=np.float64)

        for i, text in enumerate(texts):
            ngrams = self._extract_ngrams(text)
            counts = Counter(ngrams)

            for ng, count in counts.items():
                if ng in self.vocabulary:
                    idx = self.vocabulary[ng]
                    tf = count
                    if self.sublinear_tf and tf > 0:
                        tf = 1 + np.log(tf)
                    matrix[i, idx] = tf * self.idf[idx]

            # Normalização L2 por documento
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm

        return matrix

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

# Escaladores e Encoders em NumPy

class StandardScaler:
    """Z-score normalization: (x - μ) / σ"""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0  # Evitar divisão por zero
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# One-Hot Encoding para labels categóricas

class OneHotEncoder:
    """One-Hot Encoding para labels categóricas."""

    def __init__(self):
        self.classes = None
        self.class_to_idx = None

    def fit(self, labels):
        """labels: array 1D de strings/inteiros."""
        self.classes = np.sort(np.unique(labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        return self

    def transform(self, labels):
        n = len(labels)
        k = len(self.classes)
        matrix = np.zeros((n, k), dtype=np.float64)
        for i, label in enumerate(labels):
            if label in self.class_to_idx:
                matrix[i, self.class_to_idx[label]] = 1.0
        return matrix

    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)

    def inverse_transform(self, one_hot_matrix):
        indices = np.argmax(one_hot_matrix, axis=1)
        return self.classes[indices]