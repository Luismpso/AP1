
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

PASTA_WIKI = r'..\data'
FICHEIRO_HUMAN = os.path.join(PASTA_WIKI, 'Human.csv')

# --- CHAVES DE API ---
CHAVE_OPENAI = ""
CHAVE_ANTHROPIC = ""
CHAVE_GEMINI = ""

VERSOES = [
    # ======== 1º: OpenAI ========
    # ('OpenAI', 'gpt-4o-mini', 'OpenAI', 'gpt-4o-mini'),  
    # ('OpenAI', 'gpt-4o', 'OpenAI', 'gpt-4o'), 

    # ======== 2º: Anthropic (API) ========
    # ('Anthropic', 'claude-3-5-haiku-20241022', 'Anthropic', 'haiku4.5'), 
    # ('Anthropic', 'claude-sonnet-4-6', 'Anthropic', 'sonnet4.6'), 

    # ======== 3º: Google Gemini (API) ========
    # ('Gemini', 'gemini-flash-latest', 'Google', 'gemini-flash'),

    # ======== 4º: Ollama local ========
    # ('Ollama', 'llama3:latest', 'Meta', 'llama3'),          
    # ('Ollama', 'llama3.1:latest', 'Meta', 'llama3.1'),
    ('Ollama', 'llama3.2:latest', 'Meta', 'llama3.2'), 
    # ('Ollama', 'gemma:latest', 'Google', 'gemma1'),     
    # ('Ollama', 'gemma2:latest', 'Google', 'gemma2'), 
    ('Ollama', 'gemma3:latest', 'Google', 'gemma3'),    
]

# 2. Funções Auxiliares
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

# 3. Gerador de Textos com IA

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
            if num_palavras >= 80:
                return texto_final
        except Exception as e:
            tqdm.write(f"🛑 Erro com '{modelo}' em '{termo}': {e}")
            time.sleep(2)
    return melhor_texto if melhor_texto else None

# 4. Fluxo Principal: Gerar Versões e Guardar CSVs

def gerar_versoes():
    # ---- Ler termos do Human.csv ----
    if not os.path.exists(FICHEIRO_HUMAN):
        print(f"❌ Human.csv não encontrado em: {FICHEIRO_HUMAN}")
        return

    df_human = pd.read_csv(FICHEIRO_HUMAN, sep=';')
    termos = df_human['Termo'].tolist()
    print(f"📖 {len(termos)} termos carregados do Human.csv\n")

    if not VERSOES:
        print("⚠️ Nenhuma versão descomentada em VERSOES.")
        return

    for i, (provedor, modelo, label, nome_csv) in enumerate(VERSOES, 1):
        ficheiro = os.path.join(PASTA_WIKI, f'{nome_csv}.csv')

        # Saltar se já existe
        if os.path.exists(ficheiro):
            df_existente = pd.read_csv(ficheiro, sep=';')
            print(f"[{i}/{len(VERSOES)}] ⏭️  {nome_csv}.csv já existe ({len(df_existente)} exemplos). A saltar...\n")
            continue

        print(f"[{i}/{len(VERSOES)}] A gerar: {nome_csv}.csv ({provedor} → {modelo})")
        dados = []

        for termo in tqdm(termos, desc=nome_csv, unit="termo"):
            texto = gerar_ai(termo, provedor, modelo)
            if texto:
                dados.append({'Termo': termo, 'Text': texto, 'Label': label})
                
                tqdm.write(f"\n✅ [{label}] {termo} ({contar_palavras(texto)} palavras):")
                tqdm.write(f"📝 {texto}\n")

        #  Guardar CSV logo que acaba esta versão 
        if dados:
            df = pd.DataFrame(dados)
            df.to_csv(ficheiro, sep=';', index=False, encoding='utf-8')
            wcs = [contar_palavras(d['Text']) for d in dados]
            print(f"\n   💾 GUARDADO: {nome_csv}.csv → {len(dados)}/{len(termos)} válidos "
                  f"(palavras: {min(wcs)}-{max(wcs)}, média: {sum(wcs)/len(wcs):.0f})\n")
        else:
            print(f"\n   ⚠️ 0 textos gerados para {nome_csv}!\n")

    for f in sorted(os.listdir(PASTA_WIKI)):
        if f.endswith('.csv'):
            caminho = os.path.join(PASTA_WIKI, f)
            df_tmp = pd.read_csv(caminho, sep=';')
            label = df_tmp['Label'].iloc[0] if 'Label' in df_tmp.columns else '?'
            print(f"  📄 {f:<25} → {len(df_tmp):>4} exemplos  (Label: {label})")

# 5. Compilador

def compilar_dataset(ficheiros, saida='dataset.csv'):
    caminho_saida = os.path.join(PASTA_WIKI, saida)
    dfs = []

    for entrada in ficheiros:
        if isinstance(entrada, tuple):
            nome, n = entrada
        else:
            nome, n = entrada, None

        caminho = os.path.join(PASTA_WIKI, f"{nome}.csv")
        if not os.path.exists(caminho):
            print(f"⚠️ Não encontrado: {caminho}")
            continue

        df = pd.read_csv(caminho, sep=';')
        if n and n < len(df):
            df = df.sample(n=n, random_state=42)

        label = df['Label'].iloc[0] if 'Label' in df.columns else '?'
        print(f"  📄 {nome}.csv → {len(df)} exemplos (Label: {label})")
        dfs.append(df)

    if not dfs:
        print("❌ Nenhum ficheiro válido.")
        return

    df_final = pd.concat(dfs, ignore_index=True)
    df_final = df_final.drop(columns=['Termo'], errors='ignore')
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    df_final.insert(0, 'ID', [f"D1-{i+1}" for i in range(len(df_final))])
    df_final.to_csv(caminho_saida, sep=';', index=False, encoding='utf-8')

    print(f"\n🎉 Dataset final: {saida} ({len(df_final)} exemplos)")
    print(f"\nDistribuição:")
    print(df_final['Label'].value_counts().to_string())

    df_final['wc'] = df_final['Text'].apply(lambda x: len(str(x).split()))
    fora = df_final[(df_final['wc'] < 80) | (df_final['wc'] > 120)]
    if len(fora) > 0:
        print(f"\n⚠️ {len(fora)} textos fora do intervalo 80-120 palavras!")

if __name__ == "__main__":
    gerar_versoes()