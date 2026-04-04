# 🤖 Deteção de Texto Gerado por IA

**UC Aprendizagem Profunda | Mestrado em Inteligência Artificial | UMinho**

Este projeto desenvolve e compara modelos de Deep Learning e LLMs para distinguir entre texto escrito por humanos e texto gerado por diferentes modelos de IA (Anthropic, Google, Meta e OpenAI), em textos curtos (80–120 palavras) de ciências naturais e tecnologia.

## 🛠️ Implementações

* **Modelo de Raiz (NumPy):** Implementação manual de DNNs e Regressão Logística, sem bibliotecas de DL. Inclui regularização (L2 e Dropout), Early Stopping e otimizador Adam. Melhor resultado: **75.37%** no teste externo.
* **Modelos PyTorch:** DNNs com BatchNorm e LeakyReLU, Embeddings treináveis, RNNs bidirecionais (BiLSTM e BiGRU), e embeddings pré-treinados (GloVe). Inclui Grid Search de hiperparâmetros. Melhor resultado: **BiGRU 76.84%** no teste externo.
* **Transformers (HuggingFace):** Fine-tuning de BERT, DistilBERT e RoBERTa com estratégias de freeze parcial e Grid Search. Melhor resultado em CV: **DistilBERT 98.16%**.
* **LLMs (Few-Shot Prompting):** Classificação direta com Claude Opus 4.6, Gemini 3.1 Pro, DeepSeek V3 e GPT. Ensemble com weighted voting calibrado automaticamente. Melhor resultado: **Claude Opus 87.33%** no dataset do docente.
* **Validação:** Stratified K-Fold (K=3) para comparação robusta + dataset de teste externo combinado (475 textos).

## 📊 Construção dos Dados

* **Geração própria** via APIs de 15 modelos específicos (GPT-3.5/4o/4o-mini/5o-mini, Gemini Pro/Flash, Gemma 1/2/3, Opus/Sonnet/Haiku, Llama 3/3.1/3.2), com prompts orientados para ciências naturais (80–120 palavras).
* **Seleção combinatória:** Produto cartesiano de todas as combinações de modelos, avaliadas com baseline LR + TF-IDF contra os exemplos do docente, para identificar quais modelos melhor aproximavam a distribuição de teste.
* **Dataset final:** ~122.000 textos equilibrados pelas 5 classes.

## 📁 Estrutura do Repositório

```
AP/
├── database/                          # Datasets e geração de dados
│   ├── archive/                       #   Datasets descarregados (fase exploratória)
│   ├── func/                          #   Scripts de processamento de dados
│   │   ├── test.py                    #     Construção do dataset de teste combinado
│   │   ├── data.py                    #     Carregamento e limpeza
│   │   ├── dataset.py                 #     Construção do dataset final
│   │   ├── extend.py                  #     Extensão com dados gerados
│   │   └── list.py                    #     Listagem de recursos
│   ├── models/                        #   Textos gerados por cada modelo de IA
│   ├── resources/                     #   Ficheiros auxiliares
│   ├── vectors/                       #   Embeddings (GloVe, etc.)
│   ├── dataset.csv                    #   Dataset principal de treino (~122k textos)
│   ├── dataset-samples.csv            #   Exemplos do professor (125 textos, com labels)
│   ├── dataset-subm1.csv              #   Textos da submissão 1 (150, sem labels)
│   ├── dataset-subm1-labels.csv       #   Labels revelados da submissão 1 (100 textos)
│   ├── dataset-subm2.csv              #   Textos da submissão 2 (150, sem labels)
│   ├── dataset-subm2-labels.csv       #   Labels revelados da submissão 2 (100 textos)
│   ├── dataset-subm3.csv              #   Textos da submissão 3 (150, sem labels)
│   ├── dataset-subm3-labels.csv       #   Labels revelados da submissão 3 (150 textos)
│   └── dataset-test.csv               #   Dataset de teste combinado (475 textos)
├── models/                            # Modelos treinados
│   ├── numpy.pkl                      #   Melhor modelo NumPy (pesos + transformers)
│   ├── pytorch.pkl                    #   Metadados do melhor modelo PyTorch
│   ├── pytorch.pth                    #   Pesos do melhor modelo PyTorch
│   ├── transformer.pkl                #   Metadados do melhor Transformer
│   └── transformer.pth                #   Pesos do melhor Transformer
├── notebooks/                         # Notebooks de análise e treino
│   ├── Data.ipynb                     #   Exploração, seleção combinatória e análise
│   ├── Numpy.ipynb                    #   Modelos NumPy (DNN + Baseline LR)
│   ├── Pytorch.ipynb                  #   Modelos PyTorch (DNN, LSTM, GRU, GloVe)
│   ├── Tranformers.ipynb              #   Transformers (BERT, DistilBERT, RoBERTa)
│   ├── LLM.ipynb                      #   Classificação few-shot com Claude Opus
│   └── Ensemble.ipynb                 #   Calibração do ensemble de 3 LLMs
├── src/                               # Código-fonte dos modelos NumPy
│   ├── activations.py                 #   Funções de ativação (ReLU, Softmax)
│   ├── layers.py                      #   Camadas (Dense, Dropout)
│   ├── logisticregression.py          #   Regressão Logística multi-classe
│   ├── losses.py                      #   Funções de custo (Cross-Entropy)
│   ├── neuralnet.py                   #   Classe NeuralNetwork (treino, avaliação)
│   ├── optimizer.py                   #   Otimizadores (SGD, Adam)
│   ├── utils.py                       #   Pipeline de features (TF-IDF, K-Fold, etc.)
│   └── vectorizers.py                 #   TF-IDF, StandardScaler, OneHotEncoder
├── Subm1/                             # Submissão 1
│   ├── subm1-g1-MIA-A.ipynb           #   Notebook — Modelo NumPy (DNN)
│   ├── subm1-g1-MIA-A.csv             #   Previsões — Modelo NumPy
│   ├── subm1-g1-MIA-B.ipynb           #   Notebook — Modelo PyTorch (DNN)
│   └── subm1-g1-MIA-B.csv             #   Previsões — Modelo PyTorch
├── Subm2/                             # Submissão 2
│   ├── subm2-g1-MIA-A.ipynb           #   Notebook — Claude Opus 4.6 Few-shot (N=30)
│   ├── subm2-g1-MIA-A.csv             #   Previsões — Claude Opus 4.6
│   ├── subm2-g1-MIA-B.ipynb           #   Notebook — Modelo PyTorch (DNNGrid)
│   └── subm2-g1-MIA-B.csv             #   Previsões — Modelo PyTorch
├── Subm3/                             # Submissão 3
│   ├── subm3-g1-MIA-A.ipynb           #   Notebook — Claude Opus 4.6 Few-shot (N=40)
│   ├── subm3-g1-MIA-A.csv             #   Previsões — Claude Opus 4.6
│   ├── subm3-g1-MIA-B.ipynb           #   Notebook — Gemini 3.1 Pro Few-shot (N=20)
│   └── subm3-g1-MIA-B.csv             #   Previsões — Gemini 3.1 Pro
├── .gitignore
├── env.yml                            # Ambiente Conda
├── presentation.md                    # Link para o vídeo da apresentação
├── report.pdf                         # Relatório
└── README.md
```

## 🗣️ Apresentação

[![Watch Video](https://img.shields.io/badge/Ver_Apresentação-blue?style=for-the-badge&logo=dropbox)](https://www.dropbox.com/scl/fi/p6rimm296856hgoxj4ju8/Apresentacao_AP.mp4?rlkey=965zuy3p44yqvdjd58o1ftyz4&st=2bnv6me5&dl=0)

## 👥 Grupo

| Nome | Email |
|------|-------|
| Luís Miguel Pereira Silva | PG60390@alunos.uminho.pt |
| Pedro Miguel Soares de Albergaria Urbano dos Reis | PG59908@alunos.uminho.pt |
| Guilherme Lobo Pinto | PG60225@alunos.uminho.pt |
| Pedro Alexandre Silva Gomes | PG60289@alunos.uminho.pt |