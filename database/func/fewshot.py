import os
import time
import re
import random
import json
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import ollama
import anthropic
from tqdm import tqdm

# 1. Configurações

PASTA_FUNC = os.path.dirname(os.path.abspath(__file__))
PASTA_DATA = os.path.abspath(os.path.join(PASTA_FUNC, '..'))
PASTA_MODELS = os.path.join(PASTA_DATA, 'models')
PASTA_RESOURCES = os.path.join(PASTA_DATA, 'resources')

FICHEIRO_TERMOS = os.path.join(PASTA_RESOURCES, 'list.txt')
FICHEIRO_LABELS = os.path.join(PASTA_DATA, 'dataset-subm1-labels.csv')

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
    },
    {
        "nome": "Conta 5",
        "api_key": "sk-usr-1b6pcydmtbqkfne5b344sahba5ca4h17xgq",
        "endpoint": "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream",
        "channel_id": "cmnab9vpmhd5khv014ztcp61j"
    },
    {
        "nome": "Conta 6",
        "api_key": "sk-usr-b93ao218mgnx6pym456kq2ojesufaszqqfg",
        "endpoint": "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream",
        "channel_id": "cmnal2fmjiapfhv01m96clfs7"
    },
    {
        "nome": "Conta 7",
        "api_key": "sk-usr-w09rexm3vmkw1ux3jlc5cbm08jsvytzcauq",
        "endpoint": "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream",
        "channel_id": "cmnal03vmiag5hv011qk74ru0"
    },
    {
        "nome": "Conta 8",
        "api_key": "sk-usr-0z13h7zmriujl5grvgysat2leu3mukztmhyc",
        "endpoint": "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream",
        "channel_id": "cmnape7viiryjhv01pqa2utns"
    },
    {
        "nome": "Conta 9",
        "api_key": "sk-usr-y0n84csbfxsw2jfrzvpjriwcel9s87x5l8m",
        "endpoint": "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream",
        "channel_id": "cmnasvpr2j8fjhv019jusn58l"
    }
]

NUM_THREADS = 3  # Contas são distribuídas automaticamente pelas threads

# Modelos a gerar com few-shot
VERSOES_FEWSHOT = [
    ('IAEdu',     'gpt-4o',                    'OpenAI',    'openai-fewshot'),
    #('Ollama',    'llama3.2:latest',           'Meta',      'meta-fewshot'),
    #('Anthropic', 'claude-haiku-4-5-20251001', 'Anthropic', 'anthropic-fewshot'),
    #('Ollama',    'gemma3:latest',             'Google',    'google-fewshot'),
]

N_EXEMPLOS_FEWSHOT = 5
N_TEXTOS_ALVO = 24567
PAUSA_IAEDU = 1
PAUSA_RATE_LIMIT = 10

# Lock para escrita thread-safe no CSV
_csv_lock = threading.Lock()

# Lock para o tqdm (evitar output misturado)
_print_lock = threading.Lock()

# 2. Funções auxiliares de texto

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
    texto_str = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '', texto_str, flags=re.IGNORECASE)
    texto_str = re.sub(r'\[.*?\]', '', texto_str)
    texto_str = re.sub(r'\(.*?\)', '', texto_str)
    texto_str = re.sub(r'\s+', ' ', texto_str)
    texto_str = re.sub(r'\s+([,;.])', r'\1', texto_str)
    texto_str = re.sub(r'\{.*?\}', '', texto_str)
    return texto_str.strip()

# 3. Carregar exemplos do professor

def carregar_exemplos_professor():
    df = pd.read_csv(FICHEIRO_LABELS, sep=';')
    df.columns = df.columns.str.strip().str.lower()

    exemplos_por_classe = {}
    for label in df['label'].unique():
        textos = df[df['label'] == label]['text'].tolist()
        exemplos_por_classe[label] = textos
        print(f"  {label}: {len(textos)} exemplos do professor")

    return exemplos_por_classe

# 4. Construir prompt few-shot

def build_fewshot_prompt(termo, label, exemplos):
    selected = random.sample(exemplos, min(N_EXEMPLOS_FEWSHOT, len(exemplos)))

    examples_block = ""
    for i, ex in enumerate(selected, 1):
        ex_truncated = ex[:500] if len(ex) > 500 else ex
        examples_block += f"Example {i}:\n{ex_truncated}\n\n"

    prompt = f"""You are generating text about scientific topics. Your writing style must closely match the examples below.

Here are {len(selected)} examples of the style you should follow:

{examples_block}
Now write a NEW text about the topic '{termo}' in the EXACT SAME STYLE as the examples above.

Rules:
- Write between 80 and 120 words
- Write in an informative, encyclopedia-like style
- Match the tone, sentence structure, and vocabulary of the examples
- Do NOT copy the examples — write original content about '{termo}'
- Do NOT use markdown, bullet points, or titles
- Output ONLY the text, nothing else"""

    return prompt

# 5. Geração — chamada IAEdu com conta FIXA (sem rotação)

def _chamar_iaedu(prompt, conta):
    """Faz um pedido à API IAEdu com uma conta específica."""
    headers = {"x-api-key": conta["api_key"]}

    novo_thread_id = str(uuid.uuid4())
    dados_utilizador = json.dumps({"name": "api_user", "role": "student"})

    payload = {
        "channel_id": (None, conta["channel_id"]),
        "message": (None, prompt),
        "thread_id": (None, novo_thread_id),
        "user_info": (None, dados_utilizador)
    }

    resp = requests.post(
        conta["endpoint"], headers=headers,
        files=payload, stream=True, timeout=60
    )

    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"

    texto_parcial = []
    hit_rate_limit = False
    uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)

    for l in resp.iter_lines():
        if not l:
            continue
        try:
            linha_json = json.loads(l.decode())
        except json.JSONDecodeError:
            continue

        tipo = linha_json.get("type", "")

        if tipo in ("start", "end", "close"):
            continue

        if tipo == "error":
            erro_msg = linha_json.get("content", "Erro desconhecido")
            if "429" in str(erro_msg) or "rate" in str(erro_msg).lower():
                hit_rate_limit = True
                break
            return None, erro_msg

        if "content" in linha_json and tipo not in ("start", "end", "error"):
            conteudo = linha_json["content"]
            if isinstance(conteudo, str):
                if uuid_pattern.search(conteudo):
                    continue
                if conteudo.strip().lower() in ("processing", ""):
                    continue
                texto_parcial.append(conteudo)
            elif isinstance(conteudo, dict) and "text" in conteudo:
                texto_parcial.append(conteudo["text"])

    if hit_rate_limit:
        return None, "RATE_LIMIT"

    texto = "".join(texto_parcial)
    return texto if texto.strip() else None, None

# 6. Geração — função principal (recebe pool de contas da thread)

def gerar_ai_fewshot(termo, provedor, modelo, exemplos_classe, contas_pool=None, pool_idx=None):
    """Gera texto com few-shot examples do professor.
    contas_pool: lista de contas IAEdu dedicadas a esta thread.
    pool_idx: lista com [int] para rotação local (mutável para manter estado).
    """
    prompt = build_fewshot_prompt(termo, provedor, exemplos_classe)
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
                # Tentar todas as contas do pool antes de pausar
                texto_iaedu = None
                contas_com_rate_limit = 0

                for _ in range(len(contas_pool)):
                    conta = contas_pool[pool_idx[0] % len(contas_pool)]
                    pool_idx[0] += 1

                    resultado, erro = _chamar_iaedu(prompt, conta)

                    if erro == "RATE_LIMIT":
                        contas_com_rate_limit += 1
                        with _print_lock:
                            tqdm.write(f"  ⏳ {conta['nome']}: rate limit, a tentar próxima...")
                        continue
                    elif erro:
                        with _print_lock:
                            tqdm.write(f"  🛑 {conta['nome']} erro: {erro}")
                        break
                    elif resultado:
                        texto_iaedu = resultado
                        break

                if texto_iaedu:
                    texto = texto_iaedu
                    time.sleep(PAUSA_IAEDU)
                elif contas_com_rate_limit >= len(contas_pool):
                    nomes = ", ".join(c['nome'] for c in contas_pool)
                    with _print_lock:
                        tqdm.write(f"  ⏳ Pool [{nomes}] toda em rate limit. Pausa {PAUSA_RATE_LIMIT}s...")
                    time.sleep(PAUSA_RATE_LIMIT)
                    continue
                else:
                    continue

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
            with _print_lock:
                tqdm.write(f"  🛑 Erro '{modelo}' em '{termo}': {e}")
            time.sleep(2)

    return melhor_texto if melhor_texto else None

# 7. Gestão de CSVs (thread-safe)

def obter_caminho_csv(nome_csv):
    return os.path.join(PASTA_MODELS, f"{nome_csv}.csv")


def ler_termos_existentes(nome_csv):
    caminho = obter_caminho_csv(nome_csv)
    if os.path.exists(caminho):
        try:
            df = pd.read_csv(caminho, sep=';')
            return set(df['Termo'].tolist())
        except:
            return set()
    return set()


def contar_linhas_csv(nome_csv):
    caminho = obter_caminho_csv(nome_csv)
    if os.path.exists(caminho):
        try:
            df = pd.read_csv(caminho, sep=';')
            return len(df)
        except:
            return 0
    return 0


def guardar_linha_csv(nome_csv, linha_dict):
    """Guarda uma linha imediatamente no CSV (append) — THREAD-SAFE."""
    with _csv_lock:
        caminho = obter_caminho_csv(nome_csv)
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        if 'Text' in linha_dict:
            uuid_pat = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)
            linha_dict['Text'] = uuid_pat.sub('', linha_dict['Text']).strip()
            linha_dict['Text'] = linha_dict['Text'].replace(';', ',')
            linha_dict['Text'] = re.sub(r'\s+', ' ', linha_dict['Text'])
        df = pd.DataFrame([linha_dict])
        header = not os.path.exists(caminho)
        df.to_csv(caminho, sep=';', index=False, mode='a', header=header, encoding='utf-8')

# 8. Worker de uma thread — processa a sua fatia de termos

def worker_thread(thread_id, contas_pool, termos, label, provedor, modelo, nome_csv, exemplos_classe, pbar, contador):
    """Cada thread processa a sua lista de termos com o seu pool de contas."""
    gerados = 0
    falhas = 0
    pool_idx = [0]  # Mutável para manter estado de rotação entre chamadas
    nomes_contas = "+".join(c['nome'] for c in contas_pool)

    for termo in termos:
        # Verificar se já atingimos o alvo global
        with contador['lock']:
            if contador['total'] >= contador['alvo']:
                break

        tentativas = 0
        while True:
            tentativas += 1
            texto = gerar_ai_fewshot(termo, provedor, modelo, exemplos_classe,
                                     contas_pool=contas_pool, pool_idx=pool_idx)

            if texto:
                guardar_linha_csv(nome_csv, {'Termo': termo, 'Text': texto, 'Label': label})
                gerados += 1

                with contador['lock']:
                    contador['total'] += 1
                    atual = contador['total']

                pbar.update(1)
                pbar.set_description(f"{label} ({contador['base'] + atual}/{N_TEXTOS_ALVO})")

                with _print_lock:
                    tqdm.write(f"  ✅ [T{thread_id}|{nomes_contas}] {termo} ({contar_palavras(texto)}w) [tent. {tentativas}]")
                    tqdm.write(f"     📝 {texto[:120]}...")
                break
            else:
                falhas += 1
                with _print_lock:
                    tqdm.write(f"  ⚠️ [T{thread_id}|{nomes_contas}] {termo} — falhou (tent. {tentativas}), retry...")
                time.sleep(3)

    return gerados, falhas

# 9. Fluxo principal

def expandir_fewshot():
    n_contas = len(CONTAS_IAEDU)
    print("=" * 60)
    print(f"📝 Geração Few-Shot PARALELA ({NUM_THREADS} threads, {n_contas} contas)")
    print("=" * 60)

    # Distribuir contas pelas threads (round-robin)
    pools_contas = [[] for _ in range(NUM_THREADS)]
    for i, conta in enumerate(CONTAS_IAEDU):
        pools_contas[i % NUM_THREADS].append(conta)

    print("\n🔑 Distribuição de contas:")
    for t_id in range(NUM_THREADS):
        nomes = ", ".join(c['nome'] for c in pools_contas[t_id])
        print(f"    Thread {t_id}: [{nomes}]")

    print("\nA carregar exemplos do professor...")
    exemplos = carregar_exemplos_professor()

    caminho_human = os.path.join(PASTA_DATA, 'human.csv')
    if not os.path.exists(caminho_human):
        print(f"❌ human.csv não encontrado: {caminho_human}")
        return

    df_human = pd.read_csv(caminho_human, sep=';')
    todos_termos = df_human['Termo'].tolist()
    print(f"\n📖 Total de termos (human.csv): {len(todos_termos)}")

    for idx, (provedor, modelo, label, nome_csv) in enumerate(VERSOES_FEWSHOT, 1):
        print(f"\n{'=' * 60}")
        print(f"[{idx}] {label} ({modelo}) → models/{nome_csv}.csv")
        print(f"    Few-shot com {N_EXEMPLOS_FEWSHOT} exemplos | {NUM_THREADS} threads paralelas")
        print(f"{'=' * 60}")

        if label not in exemplos:
            print(f"⚠️ Sem exemplos do professor para {label}! A saltar.")
            continue

        exemplos_classe = exemplos[label]
        print(f"  📌 {len(exemplos_classe)} exemplos disponíveis para few-shot")

        n_atual = contar_linhas_csv(nome_csv)
        n_faltam = N_TEXTOS_ALVO - n_atual

        if n_faltam <= 0:
            print(f"✅ {label}: Já tem {n_atual}/{N_TEXTOS_ALVO} textos!")
            continue

        termos_feitos = ler_termos_existentes(nome_csv)
        termos_disponiveis = [t for t in todos_termos if t not in termos_feitos]

        if len(termos_disponiveis) == 0:
            print(f"⚠️ Sem termos disponíveis para {label}!")
            continue

        # Limitar ao que falta
        termos_a_processar = termos_disponiveis[:n_faltam]

        print(f"  📊 Atual: {n_atual}/{N_TEXTOS_ALVO} | Faltam: {n_faltam}")
        print(f"  🧵 A dividir {len(termos_a_processar)} termos por {NUM_THREADS} threads...")

        # Dividir termos em fatias para cada thread (round-robin para balanceamento)
        fatias = [[] for _ in range(NUM_THREADS)]
        for i, termo in enumerate(termos_a_processar):
            fatias[i % NUM_THREADS].append(termo)

        for t_id in range(NUM_THREADS):
            nomes = ", ".join(c['nome'] for c in pools_contas[t_id])
            print(f"    Thread {t_id}: {len(fatias[t_id])} termos → [{nomes}]")

        # Contador partilhado thread-safe
        contador = {
            'total': 0,
            'alvo': n_faltam,
            'base': n_atual,
            'lock': threading.Lock()
        }

        pbar = tqdm(total=n_faltam, desc=f"{label} ({n_atual}/{N_TEXTOS_ALVO})", unit="texto")

        # Lançar threads
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = []
            for t_id in range(NUM_THREADS):
                if not fatias[t_id]:
                    continue
                f = executor.submit(
                    worker_thread,
                    t_id, pools_contas[t_id], fatias[t_id],
                    label, provedor, modelo, nome_csv,
                    exemplos_classe, pbar, contador
                )
                futures.append(f)

            # Esperar por todas as threads
            total_gerados = 0
            total_falhas = 0
            for f in as_completed(futures):
                try:
                    g, fl = f.result()
                    total_gerados += g
                    total_falhas += fl
                except Exception as e:
                    print(f"  🛑 Thread falhou: {e}")

        pbar.close()

        # Reordenar CSV pela ordem do human.csv
        caminho_csv = obter_caminho_csv(nome_csv)
        if os.path.exists(caminho_csv):
            df_resultado = pd.read_csv(caminho_csv, sep=';')
            ordem_termos = {termo: i for i, termo in enumerate(todos_termos)}
            df_resultado['_ordem'] = df_resultado['Termo'].map(ordem_termos)
            df_resultado = df_resultado.sort_values('_ordem').drop(columns=['_ordem']).reset_index(drop=True)
            df_resultado.to_csv(caminho_csv, sep=';', index=False, encoding='utf-8')
            print(f"  🔄 CSV reordenado pela ordem do human.csv")

        total_final = contar_linhas_csv(nome_csv)
        print(f"💾 {nome_csv}.csv → {total_final} total (+{total_gerados} novos, {total_falhas} falhas)")

    # Resumo
    print(f"\n{'=' * 60}")
    print("📊 RESUMO — Datasets Few-Shot")
    print(f"{'=' * 60}")
    if os.path.exists(PASTA_MODELS):
        for f in sorted(os.listdir(PASTA_MODELS)):
            if f.endswith('.csv') and 'fewshot' in f:
                caminho = os.path.join(PASTA_MODELS, f)
                df_tmp = pd.read_csv(caminho, sep=';')
                print(f"  📄 models/{f:<30} → {len(df_tmp):>5} exemplos")


if __name__ == "__main__":
    expandir_fewshot()