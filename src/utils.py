import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


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
    Extrai features estilisticas do texto ORIGINAL (antes de limpeza).
    Cada modelo de IA tem padroes estilisticos diferentes:
    - comprimento de frases, diversidade de vocabulario, uso de pontuacao, etc.
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
            n_chars,                                                  # comprimento total
            n_words,                                                  # total palavras
            np.mean(word_lengths),                                    # media comprimento palavra
            np.std(word_lengths) if len(word_lengths) > 1 else 0,    # variacao comprimento palavra
            len(set(words)) / n_words,                                # type-token ratio (diversidade)
            n_sentences,                                              # total frases
            np.mean(sent_lengths),                                    # media palavras por frase
            np.std(sent_lengths) if len(sent_lengths) > 1 else 0,    # variacao palavras por frase
            sum(1 for c in text if c in '.,;:!?-') / n_chars,        # ratio pontuacao
            sum(1 for c in text if c.isdigit()) / n_chars,            # ratio digitos
            sum(1 for c in text if c.isupper()) / n_chars,            # ratio maiusculas
            text.count(',') / n_words,                                # virgulas por palavra
            text.count('.') / n_sentences,                            # pontos por frase
        ]
        features.append(feat)

    return np.array(features, dtype=np.float64)


def prepare_text_data(texts, labels, max_features=3000, char_features=3000,
                      test_size=0.2, val_size=0.1, random_state=42,
                      use_stratify=True, ngram_range=(1, 2)):
    """
    Pipeline combinado: Word TF-IDF + Char TF-IDF + Features Estilisticas.
    """
    texts_clean = [clean_text(t) for t in texts]

    # 1. Word TF-IDF (captura semantica)
    word_vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    X_word = word_vectorizer.fit_transform(texts_clean).toarray()

    # 2. Char TF-IDF (captura padroes de caracteres de cada modelo AI)
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        max_features=char_features,
        sublinear_tf=True,
        min_df=2,
        max_df=0.98,
    )
    X_char = char_vectorizer.fit_transform(texts_clean).toarray()

    # 3. Features estilisticas (do texto ORIGINAL, nao limpo)
    X_style = extract_stylistic_features(texts)
    scaler = StandardScaler()
    X_style = scaler.fit_transform(X_style)

    # 4. Combinar todas as features
    X = np.hstack([X_word, X_char, X_style])

    # 5. One-Hot Encoding das labels
    encoder = OneHotEncoder(sparse_output=False)
    labels_2d = np.array(labels).reshape(-1, 1)
    Y = encoder.fit_transform(labels_2d)

    # 6. Divisao dos dados
    stratify_param = Y if use_stratify else None
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=(test_size + val_size), random_state=random_state, stratify=stratify_param
    )

    ratio = test_size / (test_size + val_size)
    stratify_param_temp = Y_temp if use_stratify else None
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=ratio, random_state=random_state, stratify=stratify_param_temp
    )

    # Guardar todos os transformadores
    transformers = {
        'word_vectorizer': word_vectorizer,
        'char_vectorizer': char_vectorizer,
        'scaler': scaler,
    }

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, transformers, encoder


def transform_new_texts(texts, transformers):
    """
    Transforma novos textos usando os transformadores ja fitted.
    Util para inferencia em textos novos (ex: dataset de validacao do professor).
    """
    texts_clean = [clean_text(t) for t in texts]
    X_word = transformers['word_vectorizer'].transform(texts_clean).toarray()
    X_char = transformers['char_vectorizer'].transform(texts_clean).toarray()
    X_style = extract_stylistic_features(texts)
    X_style = transformers['scaler'].transform(X_style)
    return np.hstack([X_word, X_char, X_style])
