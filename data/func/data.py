import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import ollama
from openai import OpenAI
import anthropic
from tqdm import tqdm
import re
from google import genai

# 1. Configurações e Variáveis Globais
PASTA_FUNC = os.path.dirname(os.path.abspath(__file__))
PASTA_DATA = os.path.abspath(os.path.join(PASTA_FUNC, '..'))
PASTA_RESOURCES = os.path.join(PASTA_DATA, 'resources')
PASTA_MODELS = os.path.join(PASTA_DATA, 'models') # Onde serão guardados os testes

# Ficheiros de entrada
FICHEIRO_HUMAN = os.path.join(PASTA_DATA, 'human.csv')
FICHEIRO_LISTA = os.path.join(PASTA_RESOURCES, 'list.txt')

# Chaves de API 
CHAVE_OPENAI = ""
CHAVE_ANTHROPIC = ""
CHAVE_GEMINI = ""

VERSOES = [
    # ======== 1º: OpenAI ========
    # ('OpenAI', 'gpt-3.5-turbo', 'OpenAI', 'gpt-3.5-turbo'),
    # ('OpenAI', 'gpt-4o-mini', 'OpenAI', 'gpt-4o-mini'),  
    # ('OpenAI', 'gpt-5o-mini', 'OpenAI', 'gpt-5o-mini'),
    # ('OpenAI', 'gpt-4o', 'OpenAI', 'gpt-4o'), 

    # ======== 2º: Anthropic (API) ========
    # ('Anthropic', 'claude-3-5-haiku-20250101', 'Anthropic', 'haiku4.5'), 
    # ('Anthropic', 'claude-sonnet-4-6', 'Anthropic', 'sonnet4.6'), 
    # ('Anthropic', 'claude-opus-4-6', 'Anthropic', 'opus4.6'),

    # ======== 3º: Google Gemini (API) ========
    # ('Gemini', 'gemini-flash-latest', 'Google', 'gemini-flash'),
    # ('Gemini', 'gemini-pro-latest', 'Google', 'gemini-pro'),

    # ======== 4º: Ollama local ========
    # ('Ollama', 'llama3:latest', 'Meta', 'llama3'),          
    # ('Ollama', 'llama3.1:latest', 'Meta', 'llama3.1'),
    ('Ollama', 'llama3.2:latest', 'Meta', 'llama3.2'), 
    # ('Ollama', 'gemma:latest', 'Google', 'gemma1'),     
    # ('Ollama', 'gemma2:latest', 'Google', 'gemma2'), 
    ('Ollama', 'gemma3:latest', 'Google', 'gemma3'),    
]

# 2. Funções Auxiliares (Limpeza e Truncagem)
def contar_palavras(texto):
    if not texto: return 0
    return len(str(texto).split())

def truncar_texto_frases(texto):
    if not texto: return ""
    frases = re.split(r'(?<=[.!?])\s+', str(texto).strip())
    texto_final = ""
    palavras_totais = 0
    for frase in frases:
        if not frase: continue
        num_palavras_frase = len(frase.split())
        if palavras_totais + num_palavras_frase > 120:
            break
        texto_final += frase + " "
        palavras_totais += num_palavras_frase
    return texto_final.strip()

def limpar_texto(texto):
    texto_str = str(texto).replace('\n', ' ').replace('\r', '').strip()
    texto_str = texto_str.replace('ⓘ', '')
    texto_str = re.sub(r'\[.*?\]', '', texto_str)
    caracteres_ipa = r'[ˈˌːɪəʊɛæɔɒʌ]'
    texto_str = re.sub(r'\/[^\/]*?' + caracteres_ipa + r'[^\/]*?\/', '', texto_str)
    texto_str = re.sub(r'\(.*?\)', '', texto_str)
    texto_str = re.sub(r'\s+', ' ', texto_str)
    texto_str = re.sub(r'\s+([,;.])', r'\1', texto_str)
    texto_str = re.sub(r'\{.*?\}', '', texto_str)
    return texto_str.strip()

# 3. Geração de Versões com IA (OpenAI, Anthropic, Gemini, Ollama)

def gerar_ai(termo, provedor, modelo):
    prompt = (
        f"Explain the scientific topic related to '{termo}'. "
        f"Write in an informative, textbook or encyclopedia style (like a Wikipedia summary). "
        f"The response MUST be strictly between 80 and 120 words. "
        f"DO NOT write an abstract. DO NOT use markdown formatting, bullet points, or titles. "
        f"Output absolutely nothing else but the sentences of the explanation."
    )
    melhor_texto = ""
    max_palavras = 0
    for tentativa in range(3):
        try:
            texto = ""
            if provedor == "OpenAI":
                cliente = OpenAI(api_key=CHAVE_OPENAI)
                resposta = cliente.chat.completions.create(
                    model=modelo,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200
                )
                texto = resposta.choices[0].message.content
            elif provedor == "Anthropic":
                cliente = anthropic.Anthropic(api_key=CHAVE_ANTHROPIC)
                resposta = cliente.messages.create(
                    model=modelo,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                texto = resposta.content[0].text
            elif provedor == "Gemini":
                cliente_gemini = genai.Client(api_key=CHAVE_GEMINI)
                resposta = cliente_gemini.models.generate_content(
                    model=modelo,
                    contents=prompt
                )
                texto = resposta.text
            elif provedor == "Ollama":
                resposta = ollama.chat(model=modelo, messages=[{'role': 'user', 'content': prompt}])
                texto = resposta['message']['content']

            texto_limpo = limpar_texto(texto)
            texto_final = truncar_texto_frases(texto_limpo)
            num_palavras = contar_palavras(texto_final)
            
            if num_palavras > max_palavras:
                melhor_texto = texto_final
                max_palavras = num_palavras
                
            if 80 <= num_palavras <= 120:
                return texto_final
                
        except Exception as e:
            tqdm.write(f"🛑 Erro com '{modelo}' em '{termo}': {e}")
            time.sleep(2)
            
    return melhor_texto if melhor_texto else None

# 4. Fluxo Principal: Gerar as Versões e Guardar

def gerar_versoes():
    # Criar pasta models se não existir
    if not os.path.exists(PASTA_MODELS):
        os.makedirs(PASTA_MODELS)
        print(f"📁 Pasta criada: {PASTA_MODELS}")

    if not os.path.exists(FICHEIRO_HUMAN):
        print(f"❌ Erro: '{FICHEIRO_HUMAN}' não encontrado.")
        return

    df_human = pd.read_csv(FICHEIRO_HUMAN, sep=';')
    termos = df_human['Termo'].tolist()
    print(f"📖 {len(termos)} termos carregados para processamento.\n")

    for i, (provedor, modelo, label, nome_csv) in enumerate(VERSOES, 1):
        ficheiro = os.path.join(PASTA_MODELS, f'{nome_csv}.csv') # Gravação em /models/

        if os.path.exists(ficheiro):
            print(f"[{i}/{len(VERSOES)}] ⏭️  {nome_csv}.csv já existe. A saltar...")
            continue

        print(f"[{i}/{len(VERSOES)}] A gerar: {nome_csv}.csv ({provedor} → {modelo})")
        dados = []

        for termo in tqdm(termos, desc=nome_csv, unit="termo"):
            texto = gerar_ai(termo, provedor, modelo)
            if texto:
                dados.append({'Termo': termo, 'Text': texto, 'Label': label})

        if dados:
            df = pd.DataFrame(dados)
            df.to_csv(ficheiro, sep=';', index=False, encoding='utf-8')
            print(f"   💾 GUARDADO: {ficheiro}\n")

if __name__ == "__main__":
    gerar_versoes()