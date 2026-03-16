import os
import time
import pandas as pd
import ollama
from tqdm import tqdm
import re

# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_DATA = r'data'

MODELO = "gemma3:latest"
LABEL = "Google"
NOME_CSV = "gemma3"

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

def gerar_ollama(termo):
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
            resposta = ollama.chat(model=MODELO, messages=[{'role': 'user', 'content': prompt}])
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
            tqdm.write(f"🛑 Erro em '{termo}': {e}")
            time.sleep(2)
    return melhor_texto if melhor_texto else None

# ============================================================
# FLUXO PRINCIPAL
# ============================================================

def preencher():
    caminho_human = os.path.join(PASTA_DATA, 'human.csv')
    df_human = pd.read_csv(caminho_human, sep=';')
    termos_human = set(df_human['Termo'].tolist())
    print(f"📖 Termos no human.csv: {len(termos_human)}")

    caminho_csv = os.path.join(PASTA_DATA, f'{NOME_CSV}.csv')
    if os.path.exists(caminho_csv):
        df_existente = pd.read_csv(caminho_csv, sep=';')
        termos_existentes = set(df_existente['Termo'].tolist())
    else:
        df_existente = pd.DataFrame(columns=['Termo', 'Text', 'Label'])
        termos_existentes = set()
    print(f"📌 Termos já no {NOME_CSV}.csv: {len(termos_existentes)}")

    termos_faltam = [t for t in termos_human if t not in termos_existentes]
    print(f"🆕 Termos em falta: {len(termos_faltam)}")

    if not termos_faltam:
        print(f"✅ Nada a fazer — {NOME_CSV}.csv já tem todos os termos!")
        return

    dados_novos = []
    for termo in tqdm(termos_faltam, desc=LABEL, unit="termo"):
        texto = gerar_ollama(termo)
        if texto:
            dados_novos.append({'Termo': termo, 'Text': texto, 'Label': LABEL})
            tqdm.write(f"\n✅ [{LABEL}] {termo} ({contar_palavras(texto)}w):")
            tqdm.write(f"📝 {texto}\n")
        else:
            tqdm.write(f"⚠️ [{LABEL}] {termo} — falhou")

    if dados_novos:
        df_novo = pd.DataFrame(dados_novos)
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
        df_final.to_csv(caminho_csv, sep=';', index=False, encoding='utf-8')
        wcs = [contar_palavras(d['Text']) for d in dados_novos]
        print(f"\n💾 {NOME_CSV}.csv → +{len(dados_novos)} novos, {len(df_final)} total")
        print(f"   (palavras: {min(wcs)}-{max(wcs)}, média: {sum(wcs)/len(wcs):.0f})")
    else:
        print("⚠️ 0 textos gerados!")

if __name__ == "__main__":
    preencher()