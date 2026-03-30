# 🤖 AI vs Human Text Detection

**UC Aprendizagem Profunda | Mestrado em Engenharia Informática | UMinho**

Este projeto visa desenvolver modelos de Deep Learning capazes de distinguir entre texto escrito por humanos e texto gerado por diferentes modelos de IA (Google, Anthropic, Meta e OpenAI).

## 🎯 Objetivos e Desafios

O sistema resolve um problema de classificação multi-classe:

* **Classes:** Anthropic, Google, Meta, OpenAI e Human.
* **Restrição Crítica:** Os modelos devem ser otimizados para pequenos textos (80 a 120 palavras) focados em áreas de ciências naturais e tecnologia.

## 🛠️ Implementações

* **Modelo de Raiz (NumPy):** Implementação manual de Redes Neuronais Profundas (DNN) e Regressão Logística, sem uso de bibliotecas de DL. Inclui regularização (L2 e Dropout), Early Stopping e otimizador Adam.
* **Modelos Avançados (PyTorch):** Exploração de arquiteturas como DNNs com BatchNorm e LeakyReLU, Embeddings treinável, RNNs bidirecionais (BiLSTM e BiGRU), e embeddings pré-treinados (GloVe). Inclui Grid Search de hiperparâmetros.
* **Transformers (HuggingFace):** Fine-tuning de modelos pré-treinados (BERT, DistilBERT, RoBERTa) com estratégias de freeze parcial e Grid Search.
* **Validação:** Stratified K-Fold (K=5) para comparação robusta dos modelos.

## 📊 Datasets Utilizados

Os dados foram compilados a partir de fontes como:

* **HuggingFace:** OpenTuringBench, HC3, ai-text-detection-pile, M4.
* **Geração própria** via APIs de LLMs (GPT-4o, Gemini, Llama, Claude, etc.) para balanceamento de classes.

## 📁 Estrutura do Repositório

```
AP/
├── database/                          # Datasets e geração de dados
│   ├── archive/                   #   Datasets descarregados (HuggingFace, Kaggle)
│   ├── func/                      #   Scripts de processamento de dados
│   │   ├── build_test.py          #     Construção do dataset de teste combinado
│   │   ├── data.py                #     Carregamento e limpeza
│   │   ├── dataset.py             #     Construção do dataset final
│   │   ├── extend.py              #     Extensão com dados gerados
│   │   └── list.py                #     Listagem de recursos
│   ├── models/                    #   Textos gerados por cada modelo de IA
│   │   ├── gemini-flash.csv       #     Google Gemini Flash
│   │   ├── gpt-4o.csv             #     OpenAI GPT-4o
│   │   ├── llama3.csv             #     Meta Llama 3
│   │   ├── opus4.6.csv            #     Anthropic Claude Opus
│   │   └── ...                    #     (+ gemma, haiku, sonnet, etc.)
│   ├── resources/                 #   Ficheiros auxiliares
│   ├── vectors/                   #   Embeddings (GloVe, etc.)
│   ├── dataset.csv                #   Dataset principal de treino
│   ├── dataset-exemplos.csv       #   Teste do professor — 1ª entrega (125 textos)
│   ├── dataset-subm1-labels.csv   #   Labels revelados da submissão 1 (100 textos)
│   ├── dataset-test.csv           #   Dataset de teste combinado (225 textos)
│   └── dataset-subm2.csv          #   Textos da submissão 2 (150, sem labels)
├── models/                        # Modelos treinados
│   ├── numpy.pkl                  #   Melhor modelo NumPy (pesos + transformers)
│   ├── pytorch.pkl                #   Metadados do melhor modelo PyTorch
│   └── pytorch.pth                #   Pesos do melhor modelo PyTorch
├── notebooks/                     # Notebooks de análise e treino
│   ├── NLP.ipynb                  #   Exploração e pré-processamento de texto
│   ├── T1.ipynb                   #   Tarefa 1 — Análise do dataset
│   ├── T2.ipynb                   #   Tarefa 2 — Modelos NumPy (DNN + Baseline LR)
│   ├── T3.ipynb                   #   Tarefa 3 — Modelos PyTorch (DNN, LSTM, GRU, GloVe)
│   └── T3.1.ipynb                 #   Tarefa 3.1 — Transformers (BERT, DistilBERT, RoBERTa)
├── src/                           # Código-fonte dos modelos NumPy
│   ├── activations.py             #   Funções de ativação (ReLU, Softmax)
│   ├── layers.py                  #   Camadas (Dense, Dropout)
│   ├── logisticregression.py      #   Regressão Logística multi-classe
│   ├── losses.py                  #   Funções de custo (Cross-Entropy)
│   ├── neuralnet.py               #   Classe NeuralNetwork (treino, avaliação)
│   ├── optimizer.py               #   Otimizadores (SGD, Adam)
│   ├── utils.py                   #   Pipeline de features (TF-IDF, K-Fold, etc.)
│   └── vectorizers.py             #   TF-IDF, StandardScaler, OneHotEncoder
├── Subm1/                         # Submissão 1
│   ├── subm1-g1-MIA-A.ipynb       #   Notebook — Modelo NumPy
│   ├── subm1-g1-MIA-A.csv         #   Previsões — Modelo NumPy
│   ├── subm1-g1-MIA-B.ipynb       #   Notebook — Modelo PyTorch
│   └── subm1-g1-MIA-B.csv         #   Previsões — Modelo PyTorch
├── Subm2/                         # Submissão 2
│   ├── subm2-g1-MIA-A.ipynb       #   Notebook — Modelo principal (Transformer)
│   ├── subm2-g1-MIA-A.csv         #   Previsões — Modelo principal
│   ├── subm2-g1-MIA-B.ipynb       #   Notebook — Modelo alternativo
│   └── subm2-g1-MIA-B.csv         #   Previsões — Modelo alternativo
├── .gitignore
├── env.yml                        # Ambiente Conda
└── README.md
```

[![Watch Video](https://img.shields.io/badge/Watch-Demo_Video-blue?style=for-the-badge&logo=dropbox)]([https://www.dropbox.com/s/your-video-link.mp4?dl=1](https://www.dropbox.com/scl/fi/p6rimm296856hgoxj4ju8/Apresentacao_AP.mp4?rlkey=965zuy3p44yqvdjd58o1ftyz4&st=2bnv6me5&dl=1))

## 👥 Grupo

| Nome | Email |
|------|-------|
| Luís Miguel Pereira Silva | PG60390@alunos.uminho.pt |
| Pedro Miguel Soares de Albergaria Urbano dos Reis | PG59908@alunos.uminho.pt |
| Guilherme Lobo Pinto | PG60225@alunos.uminho.pt |
| Pedro Alexandre Silva Gomes | PG60289@alunos.uminho.pt |
