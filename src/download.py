import os
import re
import glob
import pandas as pd
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore') 

# As novas classes oficiais do projeto
CLASSES_ALVO = ['human', 'openai', 'meta', 'google', 'anthropic']

def processar_texto(texto):
    """Limpa o texto, resolve listas e aplica a Regra das 80-120 Palavras do Professor"""
    # 1. Resolver listas (ex: do dataset HC3)
    if isinstance(texto, (list, tuple)):
        texto = texto[0] if len(texto) > 0 else ""
        
    if pd.isna(texto) or not isinstance(texto, str): 
        return ""
        
    # Limpar espaços extra
    texto = re.sub(r'\s+', ' ', str(texto)).strip()
    palavras = texto.split()
    
    # 2. Regra Implacável: Tem de ter no mínimo 80 palavras
    if len(palavras) < 80: 
        return ""
        
    # 3. Regra Implacável: Se tiver mais de 120, cortamos para o "Ponto Ideal" (100)
    if len(palavras) > 120:
        palavras = palavras[:100]
        
    texto_final = " ".join(palavras)
    
    # Garantir que a frase acaba com pontuação para não parecer cortada a meio
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
        texto = processar_texto(row.get(col_texto, ''))
        if not texto: continue 
        
        valor_label = str(row.get(col_label, '')).lower()
        if str(classe_humano).lower() in valor_label or valor_label == '0': 
            textos_human.append(texto)
        elif str(classe_ia).lower() in valor_label or valor_label == '1': 
            textos_ia.append(texto)
            
    return textos_human, textos_ia


if __name__ == "__main__":
    os.makedirs('../data', exist_ok=True)
    dados_por_modelo = {classe: [] for classe in CLASSES_ALVO}
    print("🚀 A construir o Dataset Definitivo (Regra de 80-120 Palavras)...\n")

    # 1. Dataset de Exemplos do Professor
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

    # 2. LMSYS Chatbot Arena 
    print("-> A extrair do LMSYS Chatbot Arena (Buscando Anthropic, OpenAI, Meta, Google)...")
    try:
        # 40 mil conversas devem ser suficientes para sacar uns bons milhares de textos
        ds_lmsys = load_dataset("lmsys/chatbot_arena_conversations", split="train[:40000]")
        for row in ds_lmsys:
            # Avaliar Modelo A
            modelo_a = str(row.get('model_a', '')).lower()
            texto_a = processar_texto(row['conversation_a'][-1]['content'])
            if texto_a:
                if 'claude' in modelo_a or 'anthropic' in modelo_a: dados_por_modelo['anthropic'].append(texto_a)
                elif 'gpt' in modelo_a or 'openai' in modelo_a: dados_por_modelo['openai'].append(texto_a)
                elif 'llama' in modelo_a or 'meta' in modelo_a: dados_por_modelo['meta'].append(texto_a)
                elif 'gemini' in modelo_a or 'palm' in modelo_a or 'google' in modelo_a: dados_por_modelo['google'].append(texto_a)

            # Avaliar Modelo B
            modelo_b = str(row.get('model_b', '')).lower()
            texto_b = processar_texto(row['conversation_b'][-1]['content'])
            if texto_b:
                if 'claude' in modelo_b or 'anthropic' in modelo_b: dados_por_modelo['anthropic'].append(texto_b)
                elif 'gpt' in modelo_b or 'openai' in modelo_b: dados_por_modelo['openai'].append(texto_b)
                elif 'llama' in modelo_b or 'meta' in modelo_b: dados_por_modelo['meta'].append(texto_b)
                elif 'gemini' in modelo_b or 'palm' in modelo_b or 'google' in modelo_b: dados_por_modelo['google'].append(texto_b)
    except Exception as e: 
        print(f"Erro no LMSYS: {e}")

    # 3. Outras Fontes: OpenTuringBench, HC3 Local, Kaggle (LLM Detect AI vs Student)
    print("-> A extrair do OpenTuringBench...")
    try:
        ds_turing = load_dataset("MLNTeam-Unical/OpenTuringBench", "in_domain", split="train[:5000]")
        for row in ds_turing:
            texto = processar_texto(row.get('text', row.get('content', '')))
            if not texto: continue
            modelo = str(row.get('model', row.get('generator', row.get('label', '')))).lower()
            if 'llama' in modelo or 'meta' in modelo: dados_por_modelo['meta'].append(texto)
            elif 'gemma' in modelo or 'google' in modelo: dados_por_modelo['google'].append(texto)
    except Exception as e: pass

    print("-> A ler HC3 Local (all.jsonl)...")
    if os.path.exists('../data/raw/all.jsonl'):
        df_hc3 = pd.read_json('../data/raw/all.jsonl', lines=True)
        for text in df_hc3.get('human_answers', []): dados_por_modelo['human'].append(processar_texto(text))
        for text in df_hc3.get('chatgpt_answers', []): dados_por_modelo['openai'].append(processar_texto(text))

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
        
        # 🎛️ Painel de controlo
        EQUILIBRAR_DADOS = True  # Deixamos no False para tu veres a magia acontecer primeiro!
        
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