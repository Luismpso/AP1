# Deteção de Texto Gerado por IA

> **UC Aprendizagem Profunda · Mestrado em Inteligência Artificial · Universidade do Minho · 2025/26**

Classificação multi-classe de textos curtos (80–120 palavras) em cinco categorias — Human, Anthropic, Google, Meta e OpenAI — usando Deep Learning e LLMs.

---

## Resultados

### Rankings da Competição

| Submissão | Modelo A | Acc. A | Modelo B | Acc. B | Ranking |
|:---------:|----------|:------:|----------|:------:|:-------:|
| 1 | DNN NumPy | 71.33% | DNN PyTorch | 68.67% | 1.º / 25 |
| 2 | Claude Opus (few-shot, N=30) | 91.33% | DNN PyTorch | 72.67% | 1.º / 24 |
| 3 | Claude Opus (few-shot, N=40) | 88.00% | Gemini 3.1 Pro (few-shot, N=20) | 83.33% | 1.º / 24 |

Os rankings completos de todos os grupos encontram-se em [`rankings.xlsx`](rankings.xlsx).

### Comparação de Abordagens

| Abordagem | Melhor Modelo | K-Fold CV | Teste |
|-----------|---------------|:---------:|:-----:|
| NumPy (from scratch) | DNN 128→64 | 94.49% | 75.37%¹ |
| PyTorch | BiGRU (h=256) | 94.72% | 76.84%¹ |
| Transformers | DistilBERT (Grid Search) | 96.15% | 77.26%¹ |
| LLM (few-shot) | Claude Opus 4.6 | — | 87.33%² |
| LLM (stacking) | MLP sobre 3 LLMs | — | 92.67%² |

¹ 475 textos teste. ² 150 textos de teste (LOO-CV).

Os modelos treinados atingem >95% em validação cruzada mas ~77% no teste externo (mudança de domínio). Um meta-classificador (stacking MLP) sobre 3 LLMs atinge 92.67%, superando tanto modelos treinados como LLMs individuais.

---

## Implementações

### Tarefa 2 — Modelos From Scratch (NumPy)

Framework modular de Deep Learning implementada inteiramente em NumPy, sem qualquer biblioteca de ML/DL:

- Camadas: Dense (He init, L2), Dropout (inverted), ReLU, Softmax
- Otimizadores: SGD com Momentum, Adam (com bias correction)
- Features: TF-IDF (word 1–2 grams + char 2–4 grams) + 13 features estilísticas = 5013 features
- Treino: Mini-batch, Early Stopping, Stratified K-Fold (K=3) sobre 122k textos
- Inclui: Bag-of-Words, TF-IDF Vectorizer, StandardScaler, OneHotEncoder — tudo from scratch

### Tarefa 3 — Modelos PyTorch

**DNNs Tabulares** — 7 variantes (Wide, Narrow, Deep, LeakyReLU, ELU, Simple, VeryDeep) + Grid Search (27 configs)

**Modelos Sequenciais:**
- Embedding + DNN (128d, 256d) com masked mean pooling
- BiLSTM e BiGRU (h=128, h=256) bidirecionais, 2 camadas
- BiLSTM + GloVe (100d, 400k vetores) — frozen e fine-tuned

**Transformers (HuggingFace):**
- BERT, DistilBERT, RoBERTa — estratégias frozen vs. partial fine-tuning
- Grid Search: 27 configs (LR × Dropout × camadas descongeladas)
- Melhor: DistilBERT (lr=5e-5, drop=0.2, unfreeze=4) → 96.15% CV, 77.26% teste
- Mixed precision (FP16), checkpoints por fold

### LLMs — Few-Shot Prompting

- Modelos: Claude Opus 4.6, Gemini 3.1 Pro, GPT-5.4, DeepSeek V3
- Ensemble: Weighted majority voting (pesos calibrados por accuracy)
- Stacking: Meta-classificador MLP (LOO-CV) sobre previsões dos 3 melhores LLMs → 92.67%

---

## Construção dos Dados

1. Geração própria via APIs de 15 modelos (GPT-3.5/4o/4o-mini/5o-mini, Gemini 2.5 Pro/Flash, Gemma 1/2/3, Opus/Sonnet/Haiku, Llama 3/3.1/3.2)
2. Textos humanos extraídos da Wikipedia (revisões pré-2021) via API, com limpeza de artefactos
3. Geração few-shot — textos gerados imitando o estilo dos exemplos do professor, com paralelismo multi-thread
4. Seleção combinatória — 180 combinações testadas com baseline LR + TF-IDF contra os exemplos do docente
5. Dataset final: ~122.000 textos equilibrados pelas 5 classes

> **Nota:** Os ficheiros de dados e modelos treinados não estão incluídos no repositório devido ao seu tamanho. Para os reproduzir, consultar a secção de reprodução abaixo.

---

## Estrutura do Repositório

```
AP/
├── database/                          # Datasets e geração de dados
│   ├── func/                          #   Scripts de processamento
│   │   ├── data.py                    #     Geração de textos (APIs + Wikipedia)
│   │   ├── extend.py                  #     Expansão incremental do dataset
│   │   ├── fewshot.py                 #     Geração few-shot (multi-thread)
│   │   ├── dataset.py                 #     Construção do dataset final
│   │   ├── test.py                    #     Construção do dataset de teste combinado
│   │   └── list.py                    #     Manutenção da lista de termos
│   ├── dataset-samples.csv            #   Exemplos do professor (125 textos)
│   └── dataset-subm{1,2,3}-labels.csv #   Labels revelados pelo professor
│
├── src/                               # Framework NumPy (from scratch)
│   ├── neuralnet.py                   #   Classe NeuralNetwork (treino, avaliação)
│   ├── layers.py                      #   Dense, Dropout
│   ├── activations.py                 #   ReLU, Softmax
│   ├── optimizer.py                   #   SGD com Momentum, Adam
│   ├── losses.py                      #   Categorical Cross-Entropy
│   ├── logisticregression.py          #   Regressão Logística multi-classe (baseline)
│   ├── vectorizers.py                 #   TF-IDF, BoW, StandardScaler, OneHotEncoder
│   └── utils.py                       #   Pipeline de features, K-Fold, split
│
├── notebooks/                         # Análise e treino
│   ├── Data.ipynb                     #   EDA + seleção combinatória (180 combos)
│   ├── Numpy.ipynb                    #   Modelos NumPy (6 modelos, K-Fold + teste)
│   ├── Pytorch.ipynb                  #   PyTorch (15 modelos + Grid Search)
│   ├── Tranformers.ipynb              #   Transformers (BERT, DistilBERT, RoBERTa)
│   ├── LLM.ipynb                      #   Few-shot com Claude Opus
│   └── Ensemble.ipynb                 #   Ensemble de 4 LLMs
│
├── Subm{1,2,3}/                       # Submissões (notebooks + CSVs)
│   ├── subm{N}-g1-MIA-A.{ipynb,csv}   #   Modelo A
│   └── subm{N}-g1-MIA-B.{ipynb,csv}   #   Modelo B
│
├── report.pdf                         # Relatório
├── rankings.xlsx                      # Rankings das submissões
├── presentation.md                    # Link para vídeo da apresentação
├── env.yml                            # Ambiente Conda
└── README.md
```

---

## Reprodução

```bash
# 1. Criar ambiente
conda env create -f env.yml
conda activate AP

# 2. Gerar dados (opcional)
cd database/func
python data.py          # Gerar textos via APIs
python extend.py        # Expandir dataset
python dataset.py       # Combinar em dataset.csv

# 3. Treinar modelos
cd ../../notebooks
jupyter notebook Numpy.ipynb       # Tarefa 2
jupyter notebook Pytorch.ipynb     # Tarefa 3
jupyter notebook Tranformers.ipynb # Tarefa 3

# 4. Submissões
cd ../Subm1
jupyter notebook subm1-g1-MIA-A.ipynb
```

---

## Apresentação

[![Watch Video](https://img.shields.io/badge/Ver_Apresentação-blue?style=for-the-badge&logo=dropbox)](https://www.dropbox.com/scl/fi/p6rimm296856hgoxj4ju8/Apresentacao_AP.mp4?rlkey=965zuy3p44yqvdjd58o1ftyz4&st=2bnv6me5&dl=0)

---

## Grupo 1 — MIA

| Nome | Nº | Email |
|------|----|-------|
| Luís Miguel Pereira Silva | PG60390 | pg60390@alunos.uminho.pt |
| Pedro Miguel S. A. Urbano dos Reis | PG59908 | pg59908@alunos.uminho.pt |
| Guilherme Lobo Pinto | PG60225 | pg60225@alunos.uminho.pt |
| Pedro Alexandre Silva Gomes | PG60289 | pg60289@alunos.uminho.pt |