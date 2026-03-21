import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import ollama
import anthropic
from tqdm import tqdm
import re
import random
import uuid
import json

# 1. Configurações de caminhos e chaves

# Localização baseada na estrutura: AP/func/extend.py
PASTA_FUNC = os.path.dirname(os.path.abspath(__file__))
PASTA_DATA = os.path.abspath(os.path.join(PASTA_FUNC, '..'))
PASTA_RESOURCES = os.path.join(PASTA_DATA, 'resources')

# Ficheiros de controlo em /resources/
FICHEIRO_TERMOS = os.path.join(PASTA_RESOURCES, 'list.txt')
FICHEIRO_FALHAS = os.path.join(PASTA_RESOURCES, 'fail.txt')

CHAVE_ANTHROPIC = "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"

CONTAS_IAEDU = [
    {
        "nome": "Conta 1",
        "api_key": "sk-usr-5gned314t8prpi6cakj0v342vreij3mzh7x",
        "endpoint": "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream",
        "channel_id": "cmmuw75o1atqyhv015ja9fmo2"
    },
    {
        "nome": "Conta 2",
        "api_key": "sk-usr-dq0sqm5wqdbxtkk2tez3oqr7p726zrfhk5u",
        "endpoint": "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream",
        "channel_id": "cmmytnq8rhus4hv01e3yjj881"
    },
    {
        "nome": "Conta 3",
        "api_key": "sk-usr-23gdi3yjieky9p4prsprwk4fattnmiwtdg5",
        "endpoint": "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream",
        "channel_id": "cmmytlxmdhunehv01w2ns6sdp"
    },
    {
        "nome": "Conta 4",
        "api_key": "sk-usr-4wm81k1mxprmejf3ywwykcq2k9667xpnsbv",
        "endpoint": "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream",
        "channel_id": "cmmz16ptfigwjhv01dckivs9u"
    }
]

# Mapeamento: (provedor, modelo_api, label_csv, nome_ficheiro_csv)
VERSOES = [
    ('Anthropic', 'claude-haiku-4-5-20251001', 'Anthropic', 'haiku4.5'),
    # ('IAEdu',     'gpt-4o',                    'OpenAI',    'gpt-4o'),
    # ('Ollama',    'gemma3:latest',             'Google',    'gemma3'), 
    # ('Ollama',    'llama3.2:latest',           'Meta',      'llama3.2'),   
]

# 2. Funções auxiliares para limpeza, contagem e truncamento de texto

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

# 3. Extração de texto histórico da Wikipedia e geração de texto por IA

def obter_wiki_historica(termo, data_limite="2021-01-01T00:00:00Z"):
    url_api = "https://en.wikipedia.org/w/api.php"
    headers = {'User-Agent': 'BotEstudante/1.0 (python-requests)'}
    try:
        time.sleep(0.5)
        params = {
            "action": "query", "prop": "revisions", "titles": termo,
            "rvlimit": 1, "rvstart": data_limite, "rvdir": "older", 
            "format": "json", "redirects": 1 
        }
        resp = requests.get(url_api, params=params, headers=headers, timeout=20).json()
        paginas = resp.get("query", {}).get("pages", {})
        id_pag = list(paginas.keys())[0]
        
        if id_pag == "-1": return None
            
        oldid = paginas[id_pag]["revisions"][0]["revid"]
        parse_params = {"action": "parse", "oldid": oldid, "prop": "text", "format": "json"}
        
        resp_texto = requests.get(url_api, params=parse_params, headers=headers, timeout=20).json()
        soup = BeautifulSoup(resp_texto["parse"]["text"]["*"], "html.parser")
        
        for lixo in soup.find_all(["math", "sup", "table", "style"]): lixo.decompose()
            
        texto_acumulado = []
        palavras = 0
        for p in soup.find_all("p"):
            txt = p.get_text().strip()
            if len(txt) > 40:
                texto_acumulado.append(txt)
                palavras += len(txt.split())
                if palavras > 130: break
                
        texto_final = truncar_texto_frases(limpar_texto(" ".join(texto_acumulado)))
        return texto_final if contar_palavras(texto_final) >= 80 else None
    except: return None

def gerar_ai(termo, provedor, modelo):
    prompt = (f"Explain the scientific topic '{termo}' in 80-120 words. Encyclopedia style. No markdown.")
    for _ in range(3):
        try:
            texto = ""
            if provedor == "Anthropic":
                client = anthropic.Anthropic(api_key=CHAVE_ANTHROPIC)
                resp = client.messages.create(model=modelo, max_tokens=200, messages=[{"role":"user","content":prompt}])
                texto = resp.content[0].text
            elif provedor == "IAEdu":
                conta = random.choice(CONTAS_IAEDU)
                headers = {"x-api-key": conta["api_key"]}
                payload = {"channel_id": (None, conta["channel_id"]), "message": (None, prompt)}
                resp = requests.post(conta["endpoint"], headers=headers, files=payload, stream=True, timeout=30)
                # ... (lógica de extração de stream do IAEdu igual à que tinhas) ...
                # Para brevidade, simplificado:
                texto = "".join([json.loads(l.decode()).get("content","") for l in resp.iter_lines() if l])
            
            final = truncar_texto_frases(limpar_texto(texto))
            if contar_palavras(final) >= 80: return final
        except: time.sleep(2)
    return None

# 4. Gestão de CSVs (Leitura e Escrita)

def ler_termos_existentes(nome_csv):
    caminho = os.path.join(PASTA_DATA, f"{nome_csv}.csv")
    if os.path.exists(caminho):
        try:
            df = pd.read_csv(caminho, sep=';')
            return set(df['Termo'].tolist())
        except: return set()
    return set()

def guardar_linha_csv(nome_csv, linha_dict):
    caminho = os.path.join(PASTA_DATA, f"{nome_csv}.csv")
    df = pd.DataFrame([linha_dict])
    header = not os.path.exists(caminho)
    df.to_csv(caminho, sep=';', index=False, mode='a', header=header, encoding='utf-8')

# 5. Fluxo Principal: Expansão do Dataset

def expandir_dataset():
    if not os.path.exists(FICHEIRO_TERMOS):
        print(f"❌ Lista não encontrada em {FICHEIRO_TERMOS}")
        return

    with open(FICHEIRO_TERMOS, 'r', encoding='utf-8') as f:
        todos_termos = [l.strip() for l in f if l.strip()]

    # FASE 1: Wikipedia (Human)
    usados_human = ler_termos_existentes('human')
    termos_wiki = [t for t in todos_termos if t not in usados_human]
    
    if termos_wiki:
        print(f"🌐 Wiki: a processar {len(termos_wiki)} termos...")
        for t in tqdm(termos_wiki):
            res = obter_wiki_historica(t)
            if res:
                guardar_linha_csv('human', {'Termo': t, 'Text': res, 'Label': 'Human'})
            else:
                with open(FICHEIRO_FALHAS, "a") as fe: fe.write(t + "\n")

    # FASE 2: IAs (baseado no que existe em human.csv)
    termos_base = ler_termos_existentes('human')

    for prov, mod, lab, nome_f in VERSOES:
        ja_feitos = ler_termos_existentes(nome_f)
        faltam = [t for t in termos_base if t not in ja_feitos]
        
        if faltam:
            print(f"🤖 {lab}: a gerar {len(faltam)} textos...")
            for t in tqdm(faltam):
                txt = gerar_ai(t, prov, mod)
                if txt:
                    guardar_linha_csv(nome_f, {'Termo': t, 'Text': txt, 'Label': lab})

if __name__ == "__main__":
    expandir_dataset()