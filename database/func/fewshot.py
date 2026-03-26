import os
import time
import re
import random
import json
import uuid

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
    }
]

# Contador global para rotação sequencial de contas
_conta_idx = 0

# Modelos a gerar com few-shot
VERSOES_FEWSHOT = [
    ('IAEdu',     'gpt-4o',                    'OpenAI',    'openai-fewshot'),
    ('Ollama',    'llama3.2:latest',           'Meta',      'meta-fewshot'),
    #('Anthropic', 'claude-haiku-4-5-20251001', 'Anthropic', 'anthropic-fewshot'),
    #('Ollama',    'gemma3:latest',             'Google',    'google-fewshot'),
]

N_EXEMPLOS_FEWSHOT = 5  # Exemplos do professor no prompt
N_TEXTOS_ALVO = 24567     # Quantos textos TOTAL no CSV (não novos)
PAUSA_IAEDU = 0        # Segundos entre pedidos à IAEdu (evitar rate limit)
PAUSA_RATE_LIMIT = 10   # Pausa maior se todas as contas IAEdu estiverem em rate limit

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
    # Remover UUIDs que possam ter vazado da API
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

# 5. Geração — provedor IAEdu (GPT-4o)

def _proxima_conta_iaedu():
    """Rotação sequencial (não aleatória) das contas IAEdu."""
    global _conta_idx
    conta = CONTAS_IAEDU[_conta_idx % len(CONTAS_IAEDU)]
    _conta_idx += 1
    return conta


def _chamar_iaedu(prompt):
    """Faz um pedido à API IAEdu com handling robusto de erros."""
    conta = _proxima_conta_iaedu()
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

    # Padrão UUID para filtrar lixo
    uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)

    for l in resp.iter_lines():
        if not l:
            continue
        try:
            linha_json = json.loads(l.decode())
        except json.JSONDecodeError:
            continue

        tipo = linha_json.get("type", "")

        # Ignorar mensagens de controlo
        if tipo in ("start", "end", "close"):
            continue

        if tipo == "error":
            erro_msg = linha_json.get("content", "Erro desconhecido")
            if "429" in str(erro_msg) or "rate" in str(erro_msg).lower():
                hit_rate_limit = True
                break
            return None, erro_msg

        # Só capturar conteúdo de texto real
        if "content" in linha_json and tipo not in ("start", "end", "error"):
            conteudo = linha_json["content"]
            if isinstance(conteudo, str):
                # Filtrar UUIDs, "Processing", e outros artefactos
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

# 6. Geração — função principal

def gerar_ai_fewshot(termo, provedor, modelo, exemplos_classe):
    """Gera texto com few-shot examples do professor."""
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
                # Tentar TODAS as contas antes de pausar
                texto_iaedu = None
                contas_com_rate_limit = 0

                for _ in range(len(CONTAS_IAEDU)):
                    resultado, erro = _chamar_iaedu(prompt)

                    if erro == "RATE_LIMIT":
                        contas_com_rate_limit += 1
                        nome_conta = CONTAS_IAEDU[(_conta_idx-1) % len(CONTAS_IAEDU)]['nome']
                        tqdm.write(f"  ⏳ {nome_conta}: rate limit, a tentar próxima conta...")
                        continue  # Tentar próxima conta imediatamente
                    elif erro:
                        tqdm.write(f"  🛑 IAEdu erro: {erro}")
                        break
                    elif resultado:
                        texto_iaedu = resultado
                        break

                if texto_iaedu:
                    texto = texto_iaedu
                    time.sleep(PAUSA_IAEDU)  # Pausa SÓ após sucesso
                elif contas_com_rate_limit >= len(CONTAS_IAEDU):
                    # TODAS as contas com rate limit — agora sim, pausar
                    tqdm.write(f"  ⏳ Todas as {len(CONTAS_IAEDU)} contas com rate limit. Pausa {PAUSA_RATE_LIMIT}s...")
                    time.sleep(PAUSA_RATE_LIMIT)
                    continue  # Retry este termo
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
            tqdm.write(f"  🛑 Erro '{modelo}' em '{termo}': {e}")
            time.sleep(2)

    return melhor_texto if melhor_texto else None

# 7. Gestão de CSVs

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
    """Guarda uma linha imediatamente no CSV (append)."""
    caminho = obter_caminho_csv(nome_csv)
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    # Limpar texto de UUIDs e artefactos antes de guardar
    if 'Text' in linha_dict:
        uuid_pat = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)
        linha_dict['Text'] = uuid_pat.sub('', linha_dict['Text']).strip()
        linha_dict['Text'] = linha_dict['Text'].replace(';', ',')  # Evitar corromper CSV
        linha_dict['Text'] = re.sub(r'\s+', ' ', linha_dict['Text'])
    df = pd.DataFrame([linha_dict])
    header = not os.path.exists(caminho)
    df.to_csv(caminho, sep=';', index=False, mode='a', header=header, encoding='utf-8')

# 8. Fluxo principal

def expandir_fewshot():
    print("=" * 60)
    print("📝 Geração Few-Shot com Exemplos do Professor")
    print("=" * 60)

    print("\nA carregar exemplos do professor...")
    exemplos = carregar_exemplos_professor()

    # Usar termos do human.csv (mesma ordem)
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
        print(f"    Few-shot com {N_EXEMPLOS_FEWSHOT} exemplos do professor")
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

        print(f"  📊 Atual: {n_atual}/{N_TEXTOS_ALVO} | Faltam: {n_faltam}")
        print(f"  🆕 A gerar até atingir {N_TEXTOS_ALVO} textos...")

        gerados = 0
        falhas_total = 0

        pbar = tqdm(total=n_faltam, desc=f"{label} ({n_atual}/{N_TEXTOS_ALVO})", unit="texto")

        for termo in termos_disponiveis:
            if gerados >= n_faltam:
                break

            # Tentar até conseguir (não sai do termo sem sucesso)
            tentativas = 0
            while True:
                tentativas += 1
                texto = gerar_ai_fewshot(termo, provedor, modelo, exemplos_classe)
                if texto:
                    guardar_linha_csv(nome_csv, {'Termo': termo, 'Text': texto, 'Label': label})
                    gerados += 1
                    pbar.update(1)
                    pbar.set_description(f"{label} ({n_atual + gerados}/{N_TEXTOS_ALVO})")
                    tqdm.write(f"  ✅ [{label}] {termo} ({contar_palavras(texto)}w) [tentativa {tentativas}]")
                    tqdm.write(f"     📝 {texto[:120]}...")
                    break
                else:
                    falhas_total += 1
                    tqdm.write(f"  ⚠️ [{label}] {termo} — falhou (tentativa {tentativas}), a tentar novamente...")
                    time.sleep(3)  # Pequena pausa antes de re-tentar

        pbar.close()
        total_final = contar_linhas_csv(nome_csv)
        print(f"💾 {nome_csv}.csv → {total_final} total (+{gerados} novos)")

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