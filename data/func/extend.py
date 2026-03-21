"""
extend.py — Expansão do dataset com textos da Wikipedia (Human) e modelos de IA.

Fase 1: Recolhe textos da Wikipedia (revisões pré-2021) para cada termo da lista
Fase 2: Gera textos com cada modelo de IA para os mesmos termos

Uso:
    python extend.py

Estrutura:
    data/
    ├── func/extend.py          (este ficheiro)
    ├── resources/list.txt      (lista de termos científicos)
    ├── resources/fail.txt      (termos que falharam)
    ├── human.csv               (textos da Wikipedia)
    └── models/                 (textos gerados por cada IA)
        ├── gemini-flash.csv
        ├── gpt-4o.csv
        └── ...
"""

import os
import time
import re
import random
import json

import pandas as pd
import requests
from bs4 import BeautifulSoup
import ollama
import anthropic
from tqdm import tqdm


# ══════════════════════════════════════
# 1. Configurações de caminhos e chaves
# ══════════════════════════════════════

PASTA_FUNC = os.path.dirname(os.path.abspath(__file__))
PASTA_DATA = os.path.abspath(os.path.join(PASTA_FUNC, '..'))
PASTA_RESOURCES = os.path.join(PASTA_DATA, 'resources')
PASTA_MODELS = os.path.join(PASTA_DATA, 'models')

FICHEIRO_TERMOS = os.path.join(PASTA_RESOURCES, 'list.txt')
FICHEIRO_FALHAS = os.path.join(PASTA_RESOURCES, 'fail.txt')

NOVOS_EXEMPLOS = 100  # Quantos termos novos por corrida

CHAVE_ANTHROPIC = os.environ.get("ANTHROPIC_API_KEY", "")

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

# (provedor, modelo_api, label_csv, nome_ficheiro_csv)
VERSOES = [
    # ('Anthropic', 'claude-haiku-4-5-20251001', 'Anthropic', 'haiku4.5'),
    # ('IAEdu',     'gpt-4o',                    'OpenAI',    'gpt-4o'),
    # ('Ollama',    'gemma3:latest',             'Google',    'gemma3'),
    ('Ollama',    'llama3.2:latest',           'Meta',      'llama3.2'),
]


# ══════════════════════════════════════
# 2. Funções auxiliares de texto
# ══════════════════════════════════════

def contar_palavras(texto):
    if not texto: return 0
    return len(str(texto).split())


def truncar_texto_frases(texto):
    """Trunca texto para no máximo 120 palavras, cortando em frases completas."""
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
    """Remove artefactos de Wikipedia/formatação."""
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


# ══════════════════════════════════════
# 3. Extratores e geradores de texto
# ══════════════════════════════════════

def obter_wiki_historica(termo, data_limite="2021-01-01T00:00:00Z"):
    """Extrai texto da Wikipedia de uma revisão anterior a data_limite."""
    url_api = "https://en.wikipedia.org/w/api.php"
    headers = {
        'User-Agent': 'BotEstudanteUniversitario/1.0 (mailto:luimpsoo@gmail.com) python-requests'
    }
    try:
        time.sleep(2)
        parametros_rev = {
            "action": "query", "prop": "revisions", "titles": termo,
            "rvlimit": 1, "rvstart": data_limite, "rvdir": "older",
            "format": "json", "redirects": 1
        }
        resp = requests.get(url_api, params=parametros_rev, headers=headers, timeout=10)

        if resp.status_code == 429:
            tqdm.write(f"⚠️ Rate limit em '{termo}', a esperar 30s...")
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

        for lixo in soup.find_all(["math", "sup", "table", "style"]):
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
    """Gera texto com um modelo de IA (Anthropic, IAEdu, Ollama)."""
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
            if provedor == "Anthropic":
                cliente = anthropic.Anthropic(api_key=CHAVE_ANTHROPIC)
                resposta = cliente.messages.create(
                    model=modelo, max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                texto = resposta.content[0].text

            elif provedor == "IAEdu":
                conta = random.choice(CONTAS_IAEDU)
                headers = {"x-api-key": conta["api_key"]}
                payload = {
                    "channel_id": (None, conta["channel_id"]),
                    "message": (None, prompt)
                }
                resp = requests.post(
                    conta["endpoint"], headers=headers,
                    files=payload, stream=True, timeout=30
                )
                texto = "".join([
                    json.loads(l.decode()).get("content", "")
                    for l in resp.iter_lines() if l
                ])

            elif provedor == "Ollama":
                resposta = ollama.chat(
                    model=modelo,
                    messages=[{'role': 'user', 'content': prompt}]
                )
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


# ══════════════════════════════════════
# 4. Gestão de CSVs
# ══════════════════════════════════════

def obter_caminho_csv(nome_csv):
    """
    Devolve o caminho correto:
      - 'human' → data/human.csv
      - IAs     → data/models/<nome>.csv
    """
    if nome_csv == 'human':
        return os.path.join(PASTA_DATA, f"{nome_csv}.csv")
    else:
        return os.path.join(PASTA_MODELS, f"{nome_csv}.csv")


def ler_termos_existentes(nome_csv):
    """Lê os termos já presentes num CSV existente."""
    caminho = obter_caminho_csv(nome_csv)
    if os.path.exists(caminho):
        try:
            df = pd.read_csv(caminho, sep=';')
            return set(df['Termo'].tolist())
        except:
            return set()
    return set()


def acrescentar_ao_csv(nome_csv, novos_dados):
    """Acrescenta linhas novas a um CSV existente (ou cria-o se não existir)."""
    caminho = obter_caminho_csv(nome_csv)
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    df_novo = pd.DataFrame(novos_dados)

    if os.path.exists(caminho):
        df_existente = pd.read_csv(caminho, sep=';')
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    df_final.to_csv(caminho, sep=';', index=False, encoding='utf-8')
    return len(df_final)


# ══════════════════════════════════════
# 5. Fluxo principal
# ══════════════════════════════════════

def expandir_dataset():
    # Carregar termos da lista
    if not os.path.exists(FICHEIRO_TERMOS):
        print(f"❌ Lista não encontrada em {FICHEIRO_TERMOS}")
        return

    with open(FICHEIRO_TERMOS, 'r', encoding='utf-8') as f:
        todos_termos = list(dict.fromkeys([l.strip() for l in f if l.strip()]))

    termos_usados = ler_termos_existentes('human')
    print(f"📖 Lista total: {len(todos_termos)} termos únicos")
    print(f"📌 Já usados (em human.csv): {len(termos_usados)}")

    termos_disponiveis = [t for t in todos_termos if t not in termos_usados]
    random.shuffle(termos_disponiveis)
    print(f"🆕 Disponíveis para expandir: {len(termos_disponiveis)}")

    if len(termos_disponiveis) == 0:
        print("✅ Wikipedia: Todos os termos da lista já foram usados!")
    else:
        if len(termos_disponiveis) < NOVOS_EXEMPLOS:
            print(f"⚠️ Só há {len(termos_disponiveis)} termos disponíveis (pediste {NOVOS_EXEMPLOS})")

        # ── FASE 1: Wikipedia (Human) ──
        print(f"\n{'='*60}")
        print(f"[Fase 1] Wikipedia — a tentar obter {NOVOS_EXEMPLOS} textos novos...")
        print(f"{'='*60}")

        dados_human = []
        pbar = tqdm(total=min(NOVOS_EXEMPLOS, len(termos_disponiveis)), desc="Wikipedia")
        for termo in termos_disponiveis:
            if len(dados_human) >= NOVOS_EXEMPLOS:
                break
            pbar.set_postfix_str(f"{termo}")
            texto = obter_wiki_historica(termo)
            if texto:
                dados_human.append({'Termo': termo, 'Text': texto, 'Label': 'Human'})
                pbar.update(1)
                tqdm.write(f"  ✅ [Human] {termo} ({contar_palavras(texto)}w)")
            else:
                # Registar falha
                with open(FICHEIRO_FALHAS, "a", encoding='utf-8') as fe:
                    fe.write(termo + "\n")
        pbar.close()

        if dados_human:
            total = acrescentar_ao_csv('human', dados_human)
            print(f"💾 human.csv atualizado → {total} exemplos no total")
        else:
            print("❌ Nenhum texto novo da Wikipedia.")

    # ── FASE 2+: Gerar com cada modelo de IA ──
    termos_base = ler_termos_existentes('human')
    if len(termos_base) == 0:
        print("⚠️ Sem termos base (human.csv vazio). A abortar Fase 2.")
        return

    for idx, (provedor, modelo, label, nome_csv) in enumerate(VERSOES, 2):
        print(f"\n{'='*60}")
        print(f"[Fase {idx}] {label} ({modelo}) → models/{nome_csv}.csv")
        print(f"{'='*60}")

        termos_ja_neste_csv = ler_termos_existentes(nome_csv)
        termos_a_gerar = [t for t in termos_base if t not in termos_ja_neste_csv]

        print(f"📊 {len(termos_ja_neste_csv)} / {len(termos_base)} concluídos.")

        if not termos_a_gerar:
            print(f"✅ {label}: Dataset completo!")
            continue

        print(f"🆕 A gerar {len(termos_a_gerar)} textos novos...")
        dados_ai = []

        for termo in tqdm(termos_a_gerar, desc=label, unit="termo"):
            texto = gerar_ai(termo, provedor, modelo)
            if texto:
                dados_ai.append({'Termo': termo, 'Text': texto, 'Label': label})
                tqdm.write(f"  ✅ [{label}] {termo} ({contar_palavras(texto)}w)")
            else:
                tqdm.write(f"  ⚠️ [{label}] {termo} — falhou após 3 tentativas")

        if dados_ai:
            total = acrescentar_ao_csv(nome_csv, dados_ai)
            wcs = [contar_palavras(d['Text']) for d in dados_ai]
            print(f"💾 {nome_csv}.csv → +{len(dados_ai)} novos, {total} total "
                  f"(palavras: {min(wcs)}-{max(wcs)}, média: {sum(wcs)/len(wcs):.0f})")
        else:
            print(f"⚠️ 0 textos gerados para {nome_csv}")

    # ── RESUMO FINAL ──
    print(f"\n{'='*60}")
    print("📊 RESUMO FINAL")
    print(f"{'='*60}")

    # human.csv
    caminho_human = obter_caminho_csv('human')
    if os.path.exists(caminho_human):
        df_h = pd.read_csv(caminho_human, sep=';')
        print(f"  📄 {'human.csv':<25} → {len(df_h):>5} exemplos")

    # models/*.csv
    if os.path.exists(PASTA_MODELS):
        for f in sorted(os.listdir(PASTA_MODELS)):
            if f.endswith('.csv'):
                caminho = os.path.join(PASTA_MODELS, f)
                df_tmp = pd.read_csv(caminho, sep=';')
                label = df_tmp['Label'].iloc[0] if 'Label' in df_tmp.columns else '?'
                print(f"  📄 {'models/' + f:<25} → {len(df_tmp):>5} exemplos  (Label: {label})")


if __name__ == "__main__":
    expandir_dataset()