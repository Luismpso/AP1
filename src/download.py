import os
import re
import glob
import pandas as pd
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore') 

# As novas classes oficiais do projeto
CLASSES_ALVO = ['human', 'openai', 'meta', 'google', 'anthropic']

# Limite para não rebentar com a RAM e acelerar o processo!
LIMITE_POR_CLASSE = 1500 

def processar_texto(texto):
    """Limpa o texto, resolve listas e aplica a Regra das 80-120 Palavras do Professor"""
    if isinstance(texto, (list, tuple)):
        texto = texto[0] if len(texto) > 0 else ""
        
    if pd.isna(texto) or not isinstance(texto, str): 
        return ""
        
    texto = re.sub(r'\s+', ' ', str(texto)).strip()
    palavras = texto.split()
    
    # Regra 1: Mínimo 80 palavras
    if len(palavras) < 80: 
        return ""
        
    # Regra 2: Se > 120, cortar no ponto ideal (100 palavras)
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
        # Para por aqui se já tivermos textos suficientes para não perder tempo
        if len(textos_human) > LIMITE_POR_CLASSE and len(textos_ia) > LIMITE_POR_CLASSE:
            break
            
        texto = processar_texto(row.get(col_texto, ''))
        if not texto: continue 
        
        valor_label = str(row.get(col_label, '')).lower()
        if str(classe_humano).lower() in valor_label or valor_label == '0': 
            if len(textos_human) < LIMITE_POR_CLASSE:
                textos_human.append(texto)
        elif str(classe_ia).lower() in valor_label or valor_label == '1': 
            if len(textos_ia) < LIMITE_POR_CLASSE:
                textos_ia.append(texto)
            
    return textos_human, textos_ia


if __name__ == "__main__":
    os.makedirs('../data', exist_ok=True)
    dados_por_modelo = {classe: [] for classe in CLASSES_ALVO}
    print("🚀 A construir o Dataset Definitivo (Regra de 80-120 Palavras)...\n")

    # =========================================================
    # 1. Dataset do Professor (Prioridade Máxima)
    # =========================================================
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

    # =========================================================
    # 2. A MINA DE OURO: LMSYS Chatbot Arena
    # =========================================================
    print("-> A extrair do LMSYS Chatbot Arena (Com limite de memória!)...")
    try:
        # Carregamos apenas 30.000 para ser rápido e não esgotar a RAM
        ds_lmsys = load_dataset("lmsys/chatbot_arena_conversations", split="train[:30000]")
        
        def extrair_resposta(linha, letra):
            chave_conv = f'conversation_{letra}'
            chave_resp = f'response_{letra}'
            if chave_conv in linha:
                conv = linha[chave_conv]
                if isinstance(conv, list) and len(conv) > 0:
                    return conv[-1].get('content', '')
            if chave_resp in linha:
                return str(linha[chave_resp])
            return ""

        for row in ds_lmsys:
            # Modelo A
            modelo_a = str(row.get('model_a', '')).lower()
            texto_a = processar_texto(extrair_resposta(row, 'a'))
            if texto_a:
                classe_a = None
                if 'claude' in modelo_a or 'anthropic' in modelo_a: classe_a = 'anthropic'
                elif 'gpt' in modelo_a or 'openai' in modelo_a: classe_a = 'openai'
                elif 'llama' in modelo_a or 'meta' in modelo_a: classe_a = 'meta'
                elif 'gemini' in modelo_a or 'palm' in modelo_a or 'google' in modelo_a: classe_a = 'google'
                
                if classe_a and len(dados_por_modelo[classe_a]) < LIMITE_POR_CLASSE:
                    dados_por_modelo[classe_a].append(texto_a)

            # Modelo B
            modelo_b = str(row.get('model_b', '')).lower()
            texto_b = processar_texto(extrair_resposta(row, 'b'))
            if texto_b:
                classe_b = None
                if 'claude' in modelo_b or 'anthropic' in modelo_b: classe_b = 'anthropic'
                elif 'gpt' in modelo_b or 'openai' in modelo_b: classe_b = 'openai'
                elif 'llama' in modelo_b or 'meta' in modelo_b: classe_b = 'meta'
                elif 'gemini' in modelo_b or 'palm' in modelo_b or 'google' in modelo_b: classe_b = 'google'
                
                if classe_b and len(dados_por_modelo[classe_b]) < LIMITE_POR_CLASSE:
                    dados_por_modelo[classe_b].append(texto_b)
                
    except Exception as e: 
        print(f"   ⚠️ Erro no LMSYS: {e}")

    # =========================================================
    # 3. OUTRAS FONTES LOCAIS (Para completar os limites)
    # =========================================================
    print("-> A ler HC3 Local (all.jsonl)...")
    if os.path.exists('../data/raw/all.jsonl'):
        df_hc3 = pd.read_json('../data/raw/all.jsonl', lines=True)
        for text in df_hc3.get('human_answers', []): 
            if len(dados_por_modelo['human']) >= LIMITE_POR_CLASSE: break
            t = processar_texto(text)
            if t: dados_por_modelo['human'].append(t)
            
        for text in df_hc3.get('chatgpt_answers', []): 
            if len(dados_por_modelo['openai']) >= LIMITE_POR_CLASSE: break
            t = processar_texto(text)
            if t: dados_por_modelo['openai'].append(t)

    print("-> A extrair do Kaggle (LLM Detect AI vs Student)...")
    h_text, ia_text = extrair_kaggle("prajwaldongre/llm-detect-ai-generated-vs-student-generated-text", '0', '1')
    for t in h_text:
        if len(dados_por_modelo['human']) < LIMITE_POR_CLASSE: dados_por_modelo['human'].append(t)
    for t in ia_text:
        if len(dados_por_modelo['openai']) < LIMITE_POR_CLASSE: dados_por_modelo['openai'].append(t)

    # =========================================================
    # 4. RESUMO E EXPORTAÇÃO
    # =========================================================
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
        
        # 🎛️ PAINEL DE CONTROLO
        EQUILIBRAR_DADOS = True  
        
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