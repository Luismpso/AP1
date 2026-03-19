import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import ollama
from openai import OpenAI
import anthropic
from google import genai
from tqdm import tqdm
import re
import random

# ============================================================
# CONFIGURAÇÕES — adapta os caminhos e chaves antes de correr
# ============================================================

PASTA_DATA = r'C:\Users\Luimp\Documents\github\AP\data'
FICHEIRO_TERMOS = os.path.join(PASTA_DATA, 'lista.txt')

CHAVE_OPENAI = ""
CHAVE_ANTHROPIC = ""
CHAVE_GEMINI = ""


NOVOS_EXEMPLOS = 500  # quantos textos novos queres acrescentar

# Mapeamento: (provedor, modelo_api, label_csv, nome_ficheiro_csv)
VERSOES = [
    ('OpenAI',    'gpt-4o',                      'OpenAI',    'gpt-4o'),
    ('Anthropic', 'claude-haiku-4-5-20251001',    'Anthropic', 'haiku4.5'),
    ('Gemini',    'gemini-flash-latest',           'Google',    'gemini-flash'),
    ('Ollama',    'llama3.1:latest',               'Meta',      'llama3.1'),
]

# ============================================================
# FUNÇÕES AUXILIARES (iguais ao teu data.py original)
# ============================================================

def guardar_linha_csv(ficheiro_csv, linha_dict):
    """Guarda uma única linha no CSV em modo append, sem reescrever o ficheiro todo."""
    caminho = os.path.join(PASTA_DATA, f"{ficheiro_csv}.csv")
    df_novo = pd.DataFrame([linha_dict])
    
    # Verifica se o ficheiro já existe para saber se precisa de escrever os cabeçalhos
    ficheiro_existe = os.path.exists(caminho)
    
    # mode='a' adiciona ao fim do ficheiro. header=not ficheiro_existe escreve o cabeçalho só se o ficheiro for novo
    df_novo.to_csv(caminho, sep=';', index=False, encoding='utf-8', mode='a', header=not ficheiro_existe)

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

# ============================================================
# EXTRATORES
# ============================================================

def obter_wiki_historica(termo, data_limite="2021-01-01T00:00:00Z"):
    url_api = "https://en.wikipedia.org/w/api.php"
    headers = {
    'User-Agent': 'BotEstudanteUniversitario/1.0 (mailto:luimpsoo@gmail.com) python-requests'
}
    try:
        time.sleep(2)
        parametros_rev = {
            "action": "query", "prop": "revisions", "titles": termo,
            "rvlimit": 1, "rvstart": data_limite, "rvdir": "older", "format": "json"
        }
        resp = requests.get(url_api, params=parametros_rev, headers=headers, timeout=10)

        if resp.status_code == 429:
            tqdm.write(f"⚠️ Rate limit em '{termo}', a esperar 10s...")
            time.sleep(30)
            resp = requests.get(url_api, params=parametros_rev, headers=headers, timeout=10)
            if resp.status_code != 200:
                return None

        elif resp.status_code != 200:
            tqdm.write(f"⚠️ Wiki erro '{termo}' ({resp.status_code})")
            return None
        dados = resp.json()
        paginas = dados.get("query", {}).get("pages", {})
        id_pagina = list(paginas.keys())[0]
        if id_pagina == "-1" or "revisions" not in paginas[id_pagina]:
            return None
        id_revisao = paginas[id_pagina]["revisions"][0]["revid"]
        parametros_texto = {"action": "parse", "oldid": id_revisao, "prop": "text", "format": "json"}
        resp_texto = requests.get(url_api, params=parametros_texto, headers=headers, timeout=10).json()
        html = resp_texto["parse"]["text"]["*"]
        soup = BeautifulSoup(html, "html.parser")
        for lixo in soup.find_all(["math", "sup"]):
            lixo.decompose()
        classes_lixo = ["infobox", "metadata", "reflist", "navbox", "reference", "toc", "mwe-math-element"]
        for lixo in soup.find_all(class_=classes_lixo):
            lixo.decompose()
        texto_acumulado = []
        palavras_totais = 0
        for p in soup.find_all("p"):
            txt = p.get_text().strip()
            if len(txt) > 30:
                texto_acumulado.append(txt)
                palavras_totais += len(txt.split())
                if palavras_totais > 120: break
        texto_limpo = limpar_texto(" ".join(texto_acumulado))
        texto_final = truncar_texto_frases(texto_limpo)
        if contar_palavras(texto_final) >= 80:
            return texto_final
    except Exception as e:
        tqdm.write(f"🛑 Erro Wiki '{termo}': {e}")
    return None

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
                    model=modelo, max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                texto = resposta.content[0].text
            elif provedor == "Gemini":
                cliente_gemini = genai.Client(api_key=CHAVE_GEMINI)
                resposta = cliente_gemini.models.generate_content(
                    model=modelo, contents=prompt
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
            tqdm.write(f"🛑 Erro '{modelo}' em '{termo}': {e}")
            time.sleep(2)
    return melhor_texto if melhor_texto else None

# ============================================================
# FUNÇÕES DE LEITURA / ESCRITA
# ============================================================

def ler_termos_existentes(ficheiro_csv):
    """Lê os termos já presentes num CSV existente."""
    caminho = os.path.join(PASTA_DATA, f"{ficheiro_csv}.csv")
    if os.path.exists(caminho):
        df = pd.read_csv(caminho, sep=';')
        return set(df['Termo'].tolist())
    return set()

def acrescentar_ao_csv(ficheiro_csv, novos_dados):
    """Acrescenta linhas novas a um CSV existente (ou cria-o se não existir)."""
    caminho = os.path.join(PASTA_DATA, f"{ficheiro_csv}.csv")
    df_novo = pd.DataFrame(novos_dados)

    if os.path.exists(caminho):
        df_existente = pd.read_csv(caminho, sep=';')
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    df_final.to_csv(caminho, sep=';', index=False, encoding='utf-8')
    return len(df_final)

# ============================================================
# FLUXO PRINCIPAL
# ============================================================

def expandir_dataset():
    # 1. Carregar termos da lista e descobrir quais já foram usados
    with open(FICHEIRO_TERMOS, 'r', encoding='utf-8') as f:
        todos_termos = list(dict.fromkeys([l.strip() for l in f if l.strip()]))

    termos_usados = ler_termos_existentes('human')
    print(f"📖 Lista total: {len(todos_termos)} termos únicos")
    print(f"📌 Já usados (em human.csv): {len(termos_usados)}")

    termos_disponiveis = [t for t in todos_termos if t not in termos_usados]
    random.shuffle(termos_disponiveis)
    
    quantidade_a_gerar = len(termos_disponiveis)
    print(f"🆕 Disponíveis para procurar na Wiki: {quantidade_a_gerar}")

    # ---- FASE 1: Wikipedia (Human) ----
    if quantidade_a_gerar > 0:
        print(f"\n{'='*60}")
        print(f"[Fase 1] Wikipedia — a procurar {quantidade_a_gerar} termos novos...")
        print(f"{'='*60}")

        pbar = tqdm(total=quantidade_a_gerar, desc="Wikipedia")
        for termo in termos_disponiveis:
            pbar.set_postfix_str(f"{termo}")
            texto = obter_wiki_historica(termo)
            
            if texto:
                # GUARDA IMEDIATAMENTE NO DISCO
                linha = {'Termo': termo, 'Text': texto, 'Label': 'Human'}
                guardar_linha_csv('human', linha)
                
                tqdm.write(f"\n✅ [Human] {termo} guardado ({contar_palavras(texto)}w)!")
                tqdm.write(f"📝 {texto}\n")
            
            pbar.update(1)
        pbar.close()
    else:
        print("✅ A base 'Human' já tem todos os termos da lista.")

    # ---- FASE 2+: Gerar com cada modelo de IA ----
    # Lemos de novo o ficheiro human.csv para saber exatamente o que serve de base para as IAs.
    # Assim, se parares o código e voltares, ele sabe continuar a partir daqui!
    termos_base_ia = ler_termos_existentes('human')

    for idx, (provedor, modelo, label, nome_csv) in enumerate(VERSOES, 2):
        print(f"\n{'='*60}")
        print(f"[Fase {idx}] {label} ({modelo}) → {nome_csv}.csv")
        print(f"{'='*60}")

        # Verificar quais termos já existem NESTE csv específico
        termos_ja_neste_csv = ler_termos_existentes(nome_csv)
        
        # A IA só deve gerar textos para os termos que existem na Wiki (termos_base_ia) 
        # MAS que ainda não existem no CSV desta IA específica
        termos_a_gerar = [t for t in termos_base_ia if t not in termos_ja_neste_csv]

        if not termos_a_gerar:
            print(f"⏭️  Todos os {len(termos_base_ia)} termos já existem em {nome_csv}.csv")
            continue

        print(f"🆕 A gerar {len(termos_a_gerar)} textos novos...")

        for termo in tqdm(termos_a_gerar, desc=label, unit="termo"):
            texto = gerar_ai(termo, provedor, modelo)
            
            if texto:
                # GUARDA IMEDIATAMENTE NO DISCO
                linha = {'Termo': termo, 'Text': texto, 'Label': label}
                guardar_linha_csv(nome_csv, linha)
                
                tqdm.write(f"\n✅ [{label}] {termo} guardado ({contar_palavras(texto)}w)!")
                tqdm.write(f"📝 {texto}\n")
            else:
                tqdm.write(f"⚠️ [{label}] {termo} — falhou após 3 tentativas.")

    # ---- RESUMO FINAL ----
    print(f"\n{'='*60}")
    print("📊 RESUMO FINAL")
    print(f"{'='*60}")
    for f in sorted(os.listdir(PASTA_DATA)):
        if f.endswith('.csv') and f != 'dataset.csv':
            caminho = os.path.join(PASTA_DATA, f)
            df_tmp = pd.read_csv(caminho, sep=';')
            label = df_tmp['Label'].iloc[0] if 'Label' in df_tmp.columns else '?'
            print(f"  📄 {f:<25} → {len(df_tmp):>4} exemplos  (Label: {label})")

if __name__ == "__main__":
    expandir_dataset()