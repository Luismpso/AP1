import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def prepare_text_data(texts, labels, max_features=512, test_size=0.2, val_size=0.1, random_state=42, use_stratify=True):
    """
    Transforma textos em matrizes TF-IDF, faz o One-Hot Encoding das labels 
    e divide os dados em Treino, Validação e Teste.
    """
    
    # 1. Extração de Features de Texto (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(texts).toarray() 
    
    # 2. Codificação das Labels (One-Hot Encoding)
    encoder = OneHotEncoder(sparse_output=False)
    labels_2d = np.array(labels).reshape(-1, 1)
    Y = encoder.fit_transform(labels_2d)
    
    # Define se usamos stratify com base no parâmetro
    stratify_param = Y if use_stratify else None
    
    # 3. Divisão dos Dados (Treino vs Resto)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=(test_size + val_size), random_state=random_state, stratify=stratify_param
    )
    
    # 4. Divisão do Resto em Validação e Teste
    ratio = test_size / (test_size + val_size)
    
    # Para a segunda divisão, o stratify só funciona se Y_temp tiver amostras suficientes
    stratify_param_temp = Y_temp if use_stratify else None
    
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=ratio, random_state=random_state, stratify=stratify_param_temp
    )
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, vectorizer, encoder