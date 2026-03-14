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
import random

# 1. Configurações e Constantes

FICHEIRO_SAIDA = r'C:\Users\Luimp\Documents\github\AP\data\dataset.csv'
FICHEIRO_TERMOS = r'C:\Users\Luimp\Documents\github\AP\data\lista.txt' 

CHAVE_OPENAI = ""
CHAVE_ANTHROPIC = ""

EXEMPLOS_POR_CLASSE = 500 

MODELOS_OLLAMA = {
    'gemma2:latest': 'Google',  
    'llama3.1:latest': 'Meta' 
}

# 2. Funções Utilitárias

def contar_palavras(texto):
    if not texto: return 0
    return len(str(texto).split())

def truncar_texto_frases(texto):
    """Corta o texto por frases completas, garantindo o limite de 120 palavras."""
    if not texto: return ""
    
    # Separa por ponto, exclamação ou interrogação
    frases = re.split(r'(?<=[.!?])\s+', str(texto).strip())
    texto_final = ""
    palavras_totais = 0
    
    for frase in frases:
        if not frase: continue
        num_palavras_frase = len(frase.split())
        
        if palavras_totais + num_palavras_frase > 120:
            break # Pára antes de ultrapassar o limite
            
        texto_final += frase + " "
        palavras_totais += num_palavras_frase
        
    return texto_final.strip()

def limpar_texto(texto):
    texto_str = str(texto).replace('\n', ' ').replace('\r', '').strip()
    
    # 1. Remover o ícone de áudio da Wiki
    texto_str = texto_str.replace('ⓘ', '')
    
    # 2. Remover TUDO o que estiver entre parênteses retos (ex: [1], [vague])
    texto_str = re.sub(r'\[.*?\]', '', texto_str)
    
    # 3. Remover a pronúncia fonética que esteja solta entre barras (ex: /ˈɛnzaɪmz/)
    caracteres_ipa = r'[ˈˌːɪəʊɛæɔɒʌ]'
    texto_str = re.sub(r'\/[^\/]*?' + caracteres_ipa + r'[^\/]*?\/', '', texto_str)
    
    # 4. Remover TUDO o que estiver entre parênteses curvos (origens do grego, plurais, etc)
    texto_str = re.sub(r'\(.*?\)', '', texto_str)
    
    # 5. Limpar espaços duplos e espaços antes de pontuação (que sobram após apagar os parênteses)
    texto_str = re.sub(r'\s+', ' ', texto_str)
    texto_str = re.sub(r'\s+([,;.])', r'\1', texto_str) # Corrige coisas como "Meiosis , is..." para "Meiosis, is..."

    # 6. Remover código LaTeX perdido ou qualquer coisa entre chavetas { }
    texto_str = re.sub(r'\{.*?\}', '', texto_str)
    
    return texto_str.strip()

def guardar_backup(lista_dados, label):
    if not lista_dados: return
    df = pd.DataFrame(lista_dados)
    caminho = r'C:\Users\Luimp\Documents\github\AP\data\\' + f"{label}.csv"
    df.to_csv(caminho, sep=';', index=False, encoding='utf-8')

def carregar_termos():
    """Lê os termos de um ficheiro .txt, conta-os e remove os repetidos."""
    if os.path.exists(FICHEIRO_TERMOS):
        with open(FICHEIRO_TERMOS, 'r', encoding='utf-8') as f:
            # 1. Lê todas as linhas e remove espaços em branco
            termos_brutos = [linha.strip() for linha in f if linha.strip()]
            
        total_original = len(termos_brutos)
        
        # 2. Remove duplicados mantendo a ordem original
        termos_unicos = list(dict.fromkeys(termos_brutos))
        total_unicos = len(termos_unicos)
        duplicados = total_original - total_unicos
        
        return termos_unicos
    else:
        print(f"⚠️ Ficheiro '{FICHEIRO_TERMOS}' não encontrado. A usar lista de emergência.")
        return ["Photosynthesis", "Mitosis", "Meiosis", "Cellular respiration", "Enzyme"]

# 3. Extratores e Geradores

def obter_wiki_historica(termo, data_limite="2021-01-01T00:00:00Z"):
    url_api = "https://en.wikipedia.org/w/api.php"
    
    # DICA: A Wikimedia prefere que ponhas o teu email no User-Agent para saberem quem és!
    headers = {'User-Agent': 'BotEstudanteUniversitario/1.0 (teu_email@gmail.com) python-requests'}
    
    try:
        # 1. PEQUENA PAUSA PARA NÃO SERMOS BLOQUEADOS POR SPAM
        time.sleep(0.2)
        
        parametros_rev = {
            "action": "query", "prop": "revisions", "titles": termo,
            "rvlimit": 1, "rvstart": data_limite, "rvdir": "older", "format": "json"
        }
        resp = requests.get(url_api, params=parametros_rev, headers=headers, timeout=10)
        
        # Se a Wiki nos bloquear, avisar no terminal!
        if resp.status_code != 200:
            tqdm.write(f"⚠️ A Wiki bloqueou o pedido para '{termo}' (Erro {resp.status_code})")
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
        
        # 1. Remove tags específicas (math para fórmulas, sup para notas de rodapé/expoentes)
        for lixo in soup.find_all(["math", "sup"]):
            lixo.decompose()
            
        # 2. Remove classes conhecidas por terem lixo ou tabelas
        classes_lixo = ["infobox", "metadata", "reflist", "navbox", "reference", "toc", "mwe-math-element"]
        for lixo in soup.find_all(class_=classes_lixo):
            lixo.decompose()
        
        texto_acumulado = []
        palavras_totais = 0
        
        for p in soup.find_all("p"):
            # O get_text() aqui já virá sem as fórmulas e notas de rodapé
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
        tqdm.write(f"🛑 Erro inesperado no termo '{termo}': {e}")
        
    return None

def gerar_ai(termo, provedor, modelo=None):
    prompt = f"Explain the scientific topic related to '{termo}'. Write in an informative, textbook or encyclopedia style (like a Wikipedia summary). The response MUST be strictly between 80 and 120 words. DO NOT write an abstract. DO NOT use markdown formatting, bullet points, or titles. Output absolutely nothing else but the sentences of the explanation."
    
    melhor_texto = ""
    max_palavras = 0
    
    for tentativa in range(3):
        try:
            texto = ""
            if provedor == "OpenAI":
                cliente = OpenAI(api_key=CHAVE_OPENAI)
                resposta = cliente.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200
                )
                texto = resposta.choices[0].message.content
                
            elif provedor == "Anthropic":
                cliente = anthropic.Anthropic(api_key=CHAVE_ANTHROPIC)
                resposta = cliente.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                texto = resposta.content[0].text
                
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
            tqdm.write(f"🛑 Erro inesperado no termo '{termo}': {e}")
            time.sleep(2)
            
    return melhor_texto if melhor_texto else None

# 4. Fluxo Principal 

def construir_dataset():
    todos_termos = carregar_termos()
    random.shuffle(todos_termos)
    dataset_final = []
    
    # Fase 1: Extrair Human (Wiki)
    print(f"\n[Fase 1/4] A caçar {EXEMPLOS_POR_CLASSE} textos válidos na Wikipedia...")
    dados_human = []
    termos_garantidos = []
    
    pbar = tqdm(total=EXEMPLOS_POR_CLASSE, desc="Wikipedia (Human)")
    for termo in todos_termos:
        if len(dados_human) >= EXEMPLOS_POR_CLASSE: 
            break 
            
        pbar.set_postfix_str(f"A testar: {termo}") 
            
        texto = obter_wiki_historica(termo)
        if texto:
            dados_human.append({'Termo': termo, 'Text': texto, 'Label': 'Human'})
            termos_garantidos.append(termo)
            pbar.update(1)
            
            # Imprimir no terminal
            tqdm.write(f"\n✅ [Human] {termo} ({contar_palavras(texto)} palavras):")
            tqdm.write(f"📝 {texto}\n")
            
    pbar.close()
    dataset_final.extend(dados_human)
    guardar_backup(dados_human, 'Human')
    
    if not termos_garantidos:
        print("❌ Não foi possível extrair nenhum texto da Wikipedia. A abortar...")
        return

    # Fase 2: OpenAI
    print("\n[Fase 2/4] A gerar com OpenAI...")
    dados_openai = []
    for termo in tqdm(termos_garantidos, desc="OpenAI", unit="termo"):
        texto = gerar_ai(termo, "OpenAI")
        if texto: 
            dados_openai.append({'Termo': termo, 'Text': texto, 'Label': 'OpenAI'})
            tqdm.write(f"\n✅ [OpenAI] {termo} ({contar_palavras(texto)} palavras):")
            tqdm.write(f"📝 {texto}\n")
            
    dataset_final.extend(dados_openai)
    guardar_backup(dados_openai, 'OpenAI')

    # Fase 3: Anthropic

    print("\n[Fase 3/4] A gerar com Anthropic...")
    dados_anthropic = []
    for termo in tqdm(termos_garantidos, desc="Anthropic", unit="termo"):
        texto = gerar_ai(termo, "Anthropic")
        if texto: 
            dados_anthropic.append({'Termo': termo, 'Text': texto, 'Label': 'Anthropic'})
            tqdm.write(f"\n✅ [Anthropic] {termo} ({contar_palavras(texto)} palavras):")
            tqdm.write(f"📝 {texto}\n")
            
    dataset_final.extend(dados_anthropic)
    guardar_backup(dados_anthropic, 'Anthropic')
    
    # Fase 4: Ollama (Google e Meta)

    for modelo, label in MODELOS_OLLAMA.items():
        print(f"\n[Fase 4] A gerar com Ollama local: {label} ({modelo})...")
        dados_ollama = []
        for termo in tqdm(termos_garantidos, desc=label, unit="termo"):
            texto = gerar_ai(termo, "Ollama", modelo)
            if texto: 
                dados_ollama.append({'Termo': termo, 'Text': texto, 'Label': label})
                tqdm.write(f"\n✅ [{label}] {termo} ({contar_palavras(texto)} palavras):")
                tqdm.write(f"📝 {texto}\n")
                
        dataset_final.extend(dados_ollama)
        guardar_backup(dados_ollama, label)

    # 5. Compilar dataser e guardar

    print("\n🧹 A compilar o dataset final...")
    df = pd.DataFrame(dataset_final)
    
    df = df.drop(columns=['Termo'], errors='ignore')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, 'ID', [f"D1-{i+1}" for i in range(len(df))])
    
    df.to_csv(FICHEIRO_SAIDA, sep=';', index=False, encoding='utf-8')
    
    print(f"\n🎉 Sucesso! Dataset guardado em '{FICHEIRO_SAIDA}'")
    print("\nDistribuição de Classes Final:")
    print(df['Label'].value_counts())

if __name__ == "__main__":
    construir_dataset()