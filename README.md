# 🤖 AI vs Human Text Detection

UC Aprendizagem Profunda | Mestrado em Engenharia Informática | UMinho 

Este projeto visa desenvolver modelos de Deep Learning capazes de distinguir entre texto escrito por humanos e texto gerado por diferentes modelos de IA (Google, Mistral, Meta e OpenAI) .

## 🎯 Objetivos e Desafios

O sistema resolve um problema de classificação multi-classe:

- Classes: ai-model1 (Google), ai-model2 (Mistral), ai-model3 (Meta), ai-model4 (OpenAI) e human .

- Restrição Crítica: Os modelos devem ser otimizados para textos curtos (100 a 120 caracteres) focados em áreas de ciências naturais e tecnologia.

## 🛠️ Implementações

1. Modelo de Raiz (Numpy): Implementação manual de Redes Neuronais Profundas (DNN) e Regressão Logística, sem uso de bibliotecas de DL. Inclui regularização e Dropout.

2. Modelos Avançados (PyTorch): Exploração de arquiteturas complexas como RNNs, LSTMs, GRUs e Transformers pré-treinados (BERT).

## 📊 Datasets Utilizados

Os dados foram compilados a partir de fontes como:

- HuggingFace: OpenTuringBench, HC3, ai-text-detection-pile.

- Geração própria via APIs de LLMs para balanceamento de classes.

## 👥 Grupo

- [Luís Miguel Pereira Silva] - [PG60390@alunos.uminho.pt]
- [Pedro Miguel Soares de Albergaria Urbano dos Reis] - [PG59908@alunos.uminho.pt]
- [Guilherme Lobo Pinto] - [PG60225@alunos.uminho.pt]
- [Pedro Alexandre Silva Gomes] - [PG60289@alunos.uminho.pt]