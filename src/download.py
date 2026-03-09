import os
import re
import glob
import pandas as pd
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore') 

CLASSES_ALVO = ['human', 'openai', 'meta', 'google', 'anthropic']

def processar_texto(texto):
    """
    Limpa o texto e garante a regra do professor: 
    Textos têm de ter rigorosamente entre 80 e 120 palavras.
    """
    if not isinstance(texto, str): return ""
    texto = re.sub(r'\s+', ' ', texto).strip()
    palavras = texto.split()
    
    if len(palavras) < 80: 
        return ""
        
    if len(palavras) > 120:
        palavras = palavras[:100]
        
    texto_final = " ".join(palavras)
    
    if not texto_final.endswith(('.', '!', '?')):
        texto_final += "."
        
    return texto_final

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
        texto = processar_texto(str(row.get(col_texto, '')))
        if not texto: continue
        
        valor_label = str(row.get(col_label, '')).lower()
        if str(classe_humano).lower() in valor_label or valor_label == '0': textos_human.append(texto)
        elif str(classe_ia).lower() in valor_label or valor_label == '1': textos_ia.append(texto)
            
    return textos_human, textos_ia


if __name__ == "__main__":
    os.makedirs('../data', exist_ok=True)
    dados_por_modelo = {classe: [] for classe in CLASSES_ALVO}
    print("🚀 A construir o Dataset com Regra de 80-120 Palavras...\n")
    print("-> A ler Dataset de Exemplos do Professor (dataset-exemplos.csv)...")
    if os.path.exists('../data/raw/dataset-exemplos.csv'):
        df_prof = pd.read_csv('../data/raw/dataset-exemplos.csv', sep=';')
        for _, row in df_prof.iterrows():
            texto = processar_texto(str(row.get('Text', '')))
            if not texto: continue
            
            lbl = str(row.get('Label', '')).lower()
            for classe in CLASSES_ALVO:
                if classe in lbl:
                    dados_por_modelo[classe].append(texto)
                    break
    else:
        print("   ⚠️ Ficheiro dataset-exemplos.csv não encontrado na pasta raw!")

    print("-> A ler HC3 Local (all.jsonl)...")
    if os.path.exists('../data/raw/all.jsonl'):
        df_hc3 = pd.read_json('../data/raw/all.jsonl', lines=True)
        for text in df_hc3['human_answers']: dados_por_modelo['human'].append(processar_texto(text))
        for text in df_hc3['chatgpt_answers']: dados_por_modelo['openai'].append(processar_texto(text))

    print("-> A extrair do OpenTuringBench (Meta, Google, Anthropic)...")
    try:
        ds_turing = load_dataset("MLNTeam-Unical/OpenTuringBench", "in_domain", split="train[:5000]")
        for row in ds_turing:
            texto = processar_texto(row.get('text', row.get('content', '')))
            if not texto: continue
            
            modelo = str(row.get('model', row.get('generator', row.get('label', '')))).lower()
            if 'llama' in modelo or 'meta' in modelo: dados_por_modelo['meta'].append(texto)
            elif 'gemma' in modelo or 'google' in modelo: dados_por_modelo['google'].append(texto)
            elif 'claude' in modelo or 'anthropic' in modelo: dados_por_modelo['anthropic'].append(texto)
    except Exception as e: print(f"Erro no OpenTuringBench: {e}")

    print("-> A extrair do Kaggle (LLM Detect AI vs Student)...")
    h_text, ia_text = extrair_kaggle("prajwaldongre/llm-detect-ai-generated-vs-student-generated-text", '0', '1')
    dados_por_modelo['human'].extend(h_text)
    dados_por_modelo['openai'].extend(ia_text)

    # 4. Resumo
    print("\n--- Resumo Bruto ---")
    dataframes_finais = []
    
    for modelo, textos in dados_por_modelo.items():
        textos_unicos = list(set(textos)) # Remove repetidos
        if len(textos_unicos) > 0:
            df_modelo = pd.DataFrame({'text': textos_unicos, 'label': modelo})
            dataframes_finais.append(df_modelo)
            print(f"✅ {modelo.upper()}: {len(textos_unicos)} frases")

    if dataframes_finais:
        df_mestre = pd.concat(dataframes_finais, ignore_index=True)
        
        # 🎛️ Painel de Controlo
        EQUILIBRAR_DADOS = False  # True = Cortar pelo mínimo | False = Guardar tudo
        
        if EQUILIBRAR_DADOS:
            tamanho_minimo = df_mestre['label'].value_counts().min()
            print(f"\n⚖️ A equilibrar todas as classes para {tamanho_minimo} frases...")
            df_final = df_mestre.groupby('label').sample(n=tamanho_minimo, random_state=42)
            df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            print("\n⚠️ A guardar o dataset com TODAS as frases (Desequilibrado)...")
            df_final = df_mestre.sample(frac=1, random_state=42).reset_index(drop=True)
        
        df_final.to_csv('../data/dataset.csv', index=False)
        print(f"🚀 SUCESSO! Dataset guardado com {len(df_final)} frases!")