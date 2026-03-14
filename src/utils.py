import numpy as np
import re
from vectorizers import TfidfVectorizer, StandardScaler, OneHotEncoder


def clean_text(text):
    """
    Limpeza de texto para TF-IDF.
    """
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s\.\,\!\?\;\:\-\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_stylistic_features(texts):
    """
    Extrai features estilísticas do texto ORIGINAL (antes de limpeza).
    """
    features = []
    for text in texts:
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        n_words = max(len(words), 1)
        n_chars = max(len(text), 1)
        n_sentences = max(len(sentences), 1)

        word_lengths = [len(w) for w in words] if words else [0]
        sent_lengths = [len(s.split()) for s in sentences] if sentences else [0]

        feat = [
            n_chars,
            n_words,
            np.mean(word_lengths),
            np.std(word_lengths) if len(word_lengths) > 1 else 0,
            len(set(words)) / n_words,
            n_sentences,
            np.mean(sent_lengths),
            np.std(sent_lengths) if len(sent_lengths) > 1 else 0,
            sum(1 for c in text if c in '.,;:!?-') / n_chars,
            sum(1 for c in text if c.isdigit()) / n_chars,
            sum(1 for c in text if c.isupper()) / n_chars,
            text.count(',') / n_words,
            text.count('.') / n_sentences,
        ]
        features.append(feat)

    return np.array(features, dtype=np.float64)


def train_test_split(X, Y, test_size=0.2, random_state=42, stratify=None):
    """
    Divisão treino/teste em NumPy puro com suporte a estratificação.
    """
    rng = np.random.RandomState(random_state)

    if stratify is not None:
        # Estratificação: manter a proporção de classes
        train_idx = []
        test_idx = []
        classes = np.unique(stratify)

        for cls in classes:
            cls_indices = np.where(np.array(stratify) == cls)[0]
            rng.shuffle(cls_indices)
            n_test = max(1, int(len(cls_indices) * test_size))
            test_idx.extend(cls_indices[:n_test])
            train_idx.extend(cls_indices[n_test:])

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
    else:
        n = len(X) if isinstance(X, list) else X.shape[0]
        indices = np.arange(n)
        rng.shuffle(indices)
        n_test = int(n * test_size)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    # Suportar tanto listas (textos) como arrays
    if isinstance(X, list):
        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]

    if isinstance(Y, list):
        Y_train = [Y[i] for i in train_idx]
        Y_test = [Y[i] for i in test_idx]
    else:
        Y_train = Y[train_idx]
        Y_test = Y[test_idx]

    return X_train, X_test, Y_train, Y_test


def prepare_text_data(texts, labels, max_features=3000, char_features=3000,
                      val_size=0.2, random_state=42, use_stratify=True,
                      ngram_range=(1, 2)):
    """
    Pipeline completo SEM sklearn: Divide PRIMEIRO, Fit APENAS NO TREINO.

    Retorna
    -------
    X_train, Y_train, X_val, Y_val, transformers, encoder
    """
    # 1. Divisão antes de qualquer transformação
    stratify_param = labels if use_stratify else None
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts, labels, test_size=val_size,
        random_state=random_state, stratify=stratify_param
    )

    texts_train_clean = [clean_text(t) for t in texts_train]
    texts_val_clean = [clean_text(t) for t in texts_val]

    # 2. Word TF-IDF (NumPy puro)
    word_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        analyzer='word',
    )
    X_word_train = word_vectorizer.fit_transform(texts_train_clean)
    X_word_val = word_vectorizer.transform(texts_val_clean)

    # 3. Char TF-IDF (NumPy puro)
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        max_features=char_features,
        sublinear_tf=True,
        min_df=2,
        max_df=0.98,
    )
    X_char_train = char_vectorizer.fit_transform(texts_train_clean)
    X_char_val = char_vectorizer.transform(texts_val_clean)

    # 4. Features estilísticas
    X_style_train_raw = extract_stylistic_features(texts_train)
    X_style_val_raw = extract_stylistic_features(texts_val)

    scaler = StandardScaler()
    X_style_train = scaler.fit_transform(X_style_train_raw)
    X_style_val = scaler.transform(X_style_val_raw)

    # 5. Combinar todas as features
    X_train = np.hstack([X_word_train, X_char_train, X_style_train])
    X_val = np.hstack([X_word_val, X_char_val, X_style_val])

    # 6. One-Hot Encoding das labels (NumPy puro)
    encoder = OneHotEncoder()
    Y_train = encoder.fit_transform(np.array(labels_train))
    Y_val = encoder.transform(np.array(labels_val))

    # Guardar transformadores para novos dados
    transformers = {
        'word_vectorizer': word_vectorizer,
        'char_vectorizer': char_vectorizer,
        'scaler': scaler,
    }

    return X_train, Y_train, X_val, Y_val, transformers, encoder


def _to_dense(X):
    """Converte para array denso se for sparse (compatível com sklearn e numpy)."""
    if hasattr(X, 'toarray'):
        return X.toarray()
    return np.array(X) if not isinstance(X, np.ndarray) else X


def transform_new_texts(texts, transformers):
    texts_clean = [clean_text(t) for t in texts]
    X_word = _to_dense(transformers['word_vectorizer'].transform(texts_clean))
    X_char = _to_dense(transformers['char_vectorizer'].transform(texts_clean))

    X_style_raw = extract_stylistic_features(texts)
    X_style = _to_dense(transformers['scaler'].transform(X_style_raw))

    return np.hstack([X_word, X_char, X_style])