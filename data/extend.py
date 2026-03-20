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

# Configurações e constantes

PASTA_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)))
FICHEIRO_TERMOS = os.path.join(PASTA_DATA, 'list.txt')
FICHEIRO_FALHAS = os.path.join(PASTA_DATA, 'fail.txt')

CHAVE_ANTHROPIC = ""

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
    # ('Anthropic', 'claude-haiku-4-5-20251001', 'Anthropic', 'haiku4.5'),
    # ('IAEdu',     'gpt-4o',                    'OpenAI',    'gpt-4o'),
    ('Ollama',    'gemma3:latest',             'Google',    'gemma3'), 
    ('Ollama',    'llama3.2:latest',           'Meta',      'llama3.2'),   
]

# Funções auxiliares

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

# Extratores

def obter_wiki_historica(termo, data_limite="2021-01-01T00:00:00Z"):
    url_api = "https://en.wikipedia.org/w/api.php"
    headers = {
        'User-Agent': 'BotEstudanteUniversitario/1.0 (mailto:luimpsoo@gmail.com) python-requests'
    }
    try:
        time.sleep(0.2) 
        
        parametros_rev = {
            "action": "query", "prop": "revisions", "titles": termo,
            "rvlimit": 1, "rvstart": data_limite, "rvdir": "older", 
            "format": "json", "redirects": 1 
        }
        resp = requests.get(url_api, params=parametros_rev, headers=headers, timeout=20)

        if resp.status_code == 429:
            tqdm.write(f"⚠️ Rate limit em '{termo}', a esperar 30s...")
            time.sleep(30)
            resp = requests.get(url_api, params=parametros_rev, headers=headers, timeout=20)
            if resp.status_code != 200: return None
        elif resp.status_code != 200:
            return None
            
        dados = resp.json()
        paginas = dados.get("query", {}).get("pages", {})
        id_pagina = list(paginas.keys())[0]
        
        if id_pagina == "-1" or "revisions" not in paginas[id_pagina]:
            return None
            
        id_revisao = paginas[id_pagina]["revisions"][0]["revid"]
        parametros_texto = {"action": "parse", "oldid": id_revisao, "prop": "text", "format": "json"}
        
        time.sleep(0.8)
        
        resp_texto = requests.get(url_api, params=parametros_texto, headers=headers, timeout=20).json()
        html = resp_texto["parse"]["text"]["*"]
        soup = BeautifulSoup(html, "html.parser")
        
        for lixo in soup.find_all(["math", "sup"]): lixo.decompose()
        classes_lixo = ["infobox", "metadata", "reflist", "navbox", "reference", "toc", "mwe-math-element"]
        for lixo in soup.find_all(class_=classes_lixo): lixo.decompose()
            
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
        
        if contar_palavras(texto_final) >= 80: return texto_final
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
            conta_atual = None # Reset a cada tentativa para o IAEdu
            
            if provedor == "IAEdu":
                conta_atual = random.choice(CONTAS_IAEDU)
                headers = {"x-api-key": conta_atual["api_key"]}
                thread_nova = str(uuid.uuid4())
                
                payload = {
                    "channel_id": (None, conta_atual["channel_id"]),
                    "thread_id": (None, thread_nova),
                    "user_info": (None, "{}"),
                    "message": (None, prompt)
                }
                
                resp = requests.post(conta_atual["endpoint"], headers=headers, files=payload, stream=True, timeout=30)
                
                texto_acumulado = []
                bateu_no_limite = False
                
                for linha in resp.iter_lines():
                    if linha:
                        linha_str = linha.decode('utf-8').strip()
                        try:
                            # O IAEdu envia JSONs puros linha a linha
                            dados = json.loads(linha_str)
                            tipo = dados.get("type", "")
                            conteudo = dados.get("content", "")
                            
                            # Se a API nos mandar parar
                            if tipo == "error":
                                if "429" in str(conteudo) or "Rate limit" in str(conteudo):
                                    bateu_no_limite = True
                                    break # Sai do ciclo de leitura
                                else:
                                    tqdm.write(f"⚠️ Erro IAEdu ({conta_atual['nome']}): {conteudo}")
                                    
                            # Se for texto da IA (ignoramos os avisos de 'start', 'done' e 'error')
                            elif tipo not in ["start", "done", "error"]:
                                # A NOSSA NOVA DEFESA: Se o conteúdo for um dicionário (dict), 
                                # tentamos tirar o texto lá de dentro. Se não der, convertemos para string.
                                if isinstance(conteudo, dict):
                                    conteudo = conteudo.get("text", conteudo.get("value", ""))
                                
                                # Só adiciona à lista se for realmente um texto válido (string)
                                if isinstance(conteudo, str) and conteudo:
                                    texto_acumulado.append(conteudo)
                                
                        except json.JSONDecodeError:
                            pass # Se a linha não for JSON válido, ignora

                if bateu_no_limite:
                    tqdm.write(f"⏳ Rate Limit na {conta_atual['nome']}! A dormir 60 segundos...")
                    time.sleep(60)
                    continue # Salta para a próxima tentativa do loop (e vai escolher uma conta aleatória nova!)

                texto = "".join(texto_acumulado).strip()
                
            elif provedor == "Anthropic":
                cliente = anthropic.Anthropic(api_key=CHAVE_ANTHROPIC)
                resposta = cliente.messages.create(
                    model=modelo, max_tokens=200,
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
            # Sistema inteligente de mensagens de erro para saberes quem falhou!
            nome_erro = conta_atual['nome'] if (provedor == "IAEdu" and conta_atual) else modelo
            tqdm.write(f"🛑 Erro '{nome_erro}' em '{termo}': {e}")
            time.sleep(2)
            
    return melhor_texto if melhor_texto else None

# Funções de leitura/escrita CSV

def ler_termos_existentes(ficheiro_csv):
    caminho = os.path.join(PASTA_DATA, f"{ficheiro_csv}.csv")
    if os.path.exists(caminho):
        df = pd.read_csv(caminho, sep=';')
        return set(df['Termo'].tolist())
    return set()

def guardar_linha_csv(ficheiro_csv, linha_dict):
    caminho = os.path.join(PASTA_DATA, f"{ficheiro_csv}.csv")
    df_novo = pd.DataFrame([linha_dict])
    ficheiro_existe = os.path.exists(caminho)
    df_novo.to_csv(caminho, sep=';', index=False, encoding='utf-8', mode='a', header=not ficheiro_existe)

# Fluxo principal

def expandir_dataset():
    with open(FICHEIRO_TERMOS, 'r', encoding='utf-8') as f:
        todos_termos = list(dict.fromkeys([l.strip() for l in f if l.strip()]))

    termos_usados = ler_termos_existentes('human')
    print(f"📖 Lista total: {len(todos_termos)} termos únicos")
    print(f"📌 Já usados (em human.csv): {len(termos_usados)}")

    termos_disponiveis = [t for t in todos_termos if t not in termos_usados]
    random.shuffle(termos_disponiveis)
    
    quantidade_a_gerar = len(termos_disponiveis)
    print(f"🆕 Disponíveis para procurar na Wiki: {quantidade_a_gerar}")

    # FASE 1: Wikipedia (Human) 
    if quantidade_a_gerar > 0:
        print(f"\n{'='*60}")
        print(f"[Fase 1] Wikipedia — a procurar {quantidade_a_gerar} termos novos...")
        print(f"{'='*60}")

        pbar = tqdm(total=quantidade_a_gerar, desc="Wikipedia")
        for termo in termos_disponiveis:
            pbar.set_postfix_str(f"{termo}")
            texto = obter_wiki_historica(termo)
            
            if texto:
                linha = {'Termo': termo, 'Text': texto, 'Label': 'Human'}
                guardar_linha_csv('human', linha)
                
                tqdm.write(f"\n✅ [Human] {termo} guardado ({contar_palavras(texto)}w):")
                tqdm.write(f"📝 {texto}\n")
            else:
                with open(FICHEIRO_FALHAS, "a", encoding="utf-8") as f_erros:
                    f_erros.write(termo + "\n")
            
            pbar.update(1)
        pbar.close()
    else:
        print("✅ A base 'Human' já tem todos os termos da lista.")

    # FASE 2+: Gerar com cada modelo de IA
    termos_base_ia = ler_termos_existentes('human')

    for idx, (provedor, modelo, label, nome_csv) in enumerate(VERSOES, 2):
        print(f"\n{'='*60}")
        print(f"[Fase {idx}] {label} ({modelo}) → {nome_csv}.csv")
        print(f"{'='*60}")

        termos_ja_neste_csv = ler_termos_existentes(nome_csv)
        termos_a_gerar = [t for t in termos_base_ia if t not in termos_ja_neste_csv]

        if not termos_a_gerar:
            print(f"⏭️  Todos os termos já existem em {nome_csv}.csv")
            continue

        print(f"🆕 A gerar {len(termos_a_gerar)} textos novos...")

        for termo in tqdm(termos_a_gerar, desc=label, unit="termo"):
            texto = gerar_ai(termo, provedor, modelo)
            
            if texto:
                linha = {'Termo': termo, 'Text': texto, 'Label': label}
                guardar_linha_csv(nome_csv, linha)
                
                tqdm.write(f"\n✅ [{label}] {termo} guardado ({contar_palavras(texto)}w):")
                tqdm.write(f"📝 {texto}\n")
            else:
                tqdm.write(f"⚠️ [{label}] {termo} — falhou após 3 tentativas.")

    # Resumo final
    print(f"\n{'='*60}")
    print("📊 RESUMO FINAL")
    print(f"{'='*60}")
    for f in sorted(os.listdir(PASTA_DATA)):
        if f.endswith('.csv') and f != 'dataset.csv':
            caminho = os.path.join(PASTA_DATA, f)
            try:
                df_tmp = pd.read_csv(caminho, sep=';')
                label = df_tmp['Label'].iloc[0] if 'Label' in df_tmp.columns else '?'
                print(f"  📄 {f:<25} → {len(df_tmp):>4} exemplos  (Label: {label})")
            except pd.errors.EmptyDataError:
                print(f"  📄 {f:<25} →    0 exemplos (Ficheiro Vazio)")

if __name__ == "__main__":
    expandir_dataset()