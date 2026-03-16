import os
import time
import pandas as pd
import anthropic
from tqdm import tqdm
import re

# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_DATA = r'C:\Users\Luimp\Documents\github\AP\data'
CHAVE_ANTHROPIC = "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" 

MODELO = "claude-haiku-4-5-20251001"
LABEL = "Anthropic"
NOME_CSV = "haiku4.5"

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

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

def gerar_haiku(termo):
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
            cliente = anthropic.Anthropic(api_key=CHAVE_ANTHROPIC)
            resposta = cliente.messages.create(
                model=MODELO, max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            texto = resposta.content[0].text
            texto_limpo = limpar_texto(texto)
            texto_final = truncar_texto_frases(texto_limpo)
            num_palavras = contar_palavras(texto_final)
            if num_palavras > max_palavras:
                melhor_texto = texto_final
                max_palavras = num_palavras
            if num_palavras >= 80:
                return texto_final
        except Exception as e:
            tqdm.write(f"🛑 Erro em '{termo}': {e}")
            time.sleep(2)
    return melhor_texto if melhor_texto else None

# ============================================================
# FLUXO PRINCIPAL
# ============================================================

def preencher_haiku():
    # 1. Ler todos os termos do human.csv
    caminho_human = os.path.join(PASTA_DATA, 'human.csv')
    df_human = pd.read_csv(caminho_human, sep=';')
    termos_human = set(df_human['Termo'].tolist())
    print(f"📖 Termos no human.csv: {len(termos_human)}")

    # 2. Ler termos já existentes no haiku4_5.csv
    caminho_haiku = os.path.join(PASTA_DATA, f'{NOME_CSV}.csv')
    if os.path.exists(caminho_haiku):
        df_haiku = pd.read_csv(caminho_haiku, sep=';')
        termos_haiku = set(df_haiku['Termo'].tolist())
    else:
        df_haiku = pd.DataFrame(columns=['Termo', 'Text', 'Label'])
        termos_haiku = set()
    print(f"📌 Termos já no {NOME_CSV}.csv: {len(termos_haiku)}")

    # 3. Calcular os que faltam
    termos_faltam = [t for t in termos_human if t not in termos_haiku]
    print(f"🆕 Termos em falta: {len(termos_faltam)}")

    if not termos_faltam:
        print("✅ Nada a fazer — haiku4_5.csv já tem todos os termos!")
        return

    # 4. Gerar
    dados_novos = []
    for termo in tqdm(termos_faltam, desc=LABEL, unit="termo"):
        texto = gerar_haiku(termo)
        if texto:
            dados_novos.append({'Termo': termo, 'Text': texto, 'Label': LABEL})
            tqdm.write(f"\n✅ [{LABEL}] {termo} ({contar_palavras(texto)}w):")
            tqdm.write(f"📝 {texto}\n")
        else:
            tqdm.write(f"⚠️ [{LABEL}] {termo} — falhou")

    # 5. Acrescentar ao CSV
    if dados_novos:
        df_novo = pd.DataFrame(dados_novos)
        df_final = pd.concat([df_haiku, df_novo], ignore_index=True)
        df_final.to_csv(caminho_haiku, sep=';', index=False, encoding='utf-8')
        print(f"\n💾 {NOME_CSV}.csv → +{len(dados_novos)} novos, {len(df_final)} total")
    else:
        print("⚠️ 0 textos gerados!")

if __name__ == "__main__":
    preencher_haiku()