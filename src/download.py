import os
import re
import glob
import pandas as pd
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore') 

CLASSES_ALVO = ['human', 'openai', 'meta', 'google', 'mistral']

def clean_and_shorten(text, target_length=110):
    if not isinstance(text, str) or len(text.strip()) < 20: return ""
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > target_length:
        shortened = text[:target_length]
        last_space = shortened.rfind(' ')
        return shortened[:last_space] + "." if last_space > 0 else shortened + "."
    return text

def processar_texto(lista_ou_str):
    if isinstance(lista_ou_str, (list, tuple)):
        return clean_and_shorten(lista_ou_str[0]) if len(lista_ou_str) > 0 else ""
    return clean_and_shorten(lista_ou_str)

def extrair_kaggle(dataset_ref, classe_humano, classe_ia):
    pasta_destino = f"../data/raw/{dataset_ref.split('/')[1]}"
    if not os.path.exists(pasta_destino): return [], []
    ficheiros_csv = glob.glob(f"{pasta_destino}/*.csv")
    if not ficheiros_csv: return [], []
        
    df = pd.read_csv(ficheiros_csv[0]) 
    textos_human, textos_ia = [], []
    col_texto = next((c for c in df.columns if any(x in c.lower() for x in ['text', 'abstract', 'content'])), None)
    col_label = next((c for c in df.columns if any(x in c.lower() for x in ['label', 'generated', 'source'])), None)
            
    for _, row in df.iterrows():
        texto = processar_texto(str(row[col_texto]))
        if len(texto) < 50: continue
        valor_label = str(row[col_label]).lower()
        if str(classe_humano).lower() in valor_label or valor_label == '0': textos_human.append(texto)
        elif str(classe_ia).lower() in valor_label or valor_label == '1': textos_ia.append(texto)
            
    return textos_human, textos_ia


if __name__ == "__main__":
    os.makedirs('../data', exist_ok=True)
    dados_por_modelo = {classe: [] for classe in CLASSES_ALVO}
    print("🚀 A recuperar e equilibrar o Dataset Multi-Classes...\n")

    # 1. Files Locais (HC3 e NicolaiSivesind)
    print("-> A ler HC3 Local (all.jsonl)...")
    if os.path.exists('../data/raw/all.jsonl'):
        df_hc3 = pd.read_json('../data/raw/all.jsonl', lines=True)
        for text in df_hc3['human_answers']: dados_por_modelo['human'].append(processar_texto(text))
        for text in df_hc3['chatgpt_answers']: dados_por_modelo['openai'].append(processar_texto(text))

    print("-> A ler NicolaiSivesind Local (research-abstracts-labeled.csv)...")
    if os.path.exists('../data/raw/research-abstracts-labeled.csv'):
        df_nic = pd.read_csv('../data/raw/research-abstracts-labeled.csv')
        for _, row in df_nic.iterrows():
            texto = processar_texto(str(row.get('text', row.get('abstract', ''))))
            if len(texto) < 50: continue
            lbl = str(row.get('label', row.get('source', ''))).lower()
            if lbl == '0' or 'human' in lbl: dados_por_modelo['human'].append(texto)
            elif lbl == '1' or 'machine' in lbl or 'ai' in lbl: dados_por_modelo['openai'].append(texto)

    # 2. HuggingFace
    print("-> A extrair do OpenTuringBench (Meta, Google, Mistral)...")
    try:
        ds_turing = load_dataset("MLNTeam-Unical/OpenTuringBench", "in_domain", split="train[:5000]")
        for row in ds_turing:
            texto_limpo = processar_texto(row.get('text', row.get('content', '')))
            if not texto_limpo or len(texto_limpo) < 50: continue
            
            modelo = str(row.get('model', row.get('generator', row.get('label', '')))).lower()
            if 'llama' in modelo or 'meta' in modelo: dados_por_modelo['meta'].append(texto_limpo)
            elif 'gemma' in modelo or 'google' in modelo: dados_por_modelo['google'].append(texto_limpo)
            elif 'mistral' in modelo: dados_por_modelo['mistral'].append(texto_limpo)
    except Exception as e: print(f"Erro no OpenTuringBench: {e}")

    # 3. Kaggle
    print("-> A extrair do Kaggle (LLM Detect AI vs Student)...")
    h_text, ia_text = extrair_kaggle("prajwaldongre/llm-detect-ai-generated-vs-student-generated-text", '0', '1')
    dados_por_modelo['human'].extend(h_text)
    dados_por_modelo['openai'].extend(ia_text)

    print("-> A extrair do Kaggle (GPT vs Human Abstracts)...")
    h_text2, ia_text2 = extrair_kaggle("heleneeriksen/gpt-vs-human-a-corpus-of-research-abstracts", 'human', 'ai')
    dados_por_modelo['human'].extend(h_text2)
    dados_por_modelo['openai'].extend(ia_text2)

    # 4. Resumo bruto e limpeza final 
    print("\nResumo:")
    dataframes_finais = []
    
    for modelo, textos in dados_por_modelo.items():
        textos_unicos = list(set([t for t in textos if len(t) > 50])) 
        if len(textos_unicos) > 0:
            df_modelo = pd.DataFrame({'text': textos_unicos, 'label': modelo})
            dataframes_finais.append(df_modelo)
            print(f"✅ {modelo.upper()}: {len(textos_unicos)} frases")

    if dataframes_finais:
        df_mestre = pd.concat(dataframes_finais, ignore_index=True)
    
        print("\n⚖️ A equilibrar o dataset para a rede neuronal...")
        tamanho_minimo = df_mestre['label'].value_counts().min()
        print(f"-> A cortar todas as classes para ficarem com {tamanho_minimo} frases exatas.")
        
        df_equilibrado = df_mestre.groupby('label').sample(n=tamanho_minimo, random_state=42)
        df_equilibrado = df_equilibrado.sample(frac=1, random_state=42).reset_index(drop=True)
        
        df_equilibrado.to_csv('../data/dataset.csv', index=False)
        print(f"\n🚀 SUCESSO! O dataset.csv final foi guardado e está perfeitamente equilibrado!")