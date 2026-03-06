import os
import re
import glob
import pandas as pd
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore') 

CLASSES_ALVO = ['human', 'openai', 'meta', 'google', 'mistral']

def clean_and_shorten(text, target_length=110):
    """Limpa o texto e corta para 100-120 caracteres."""
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

def extrair_kaggle(dataset_ref, classe_humano, classe_ia, label_ia='openai'):
    """Faz o download de um dataset do Kaggle, extrai e separa as classes."""
    import kaggle
    print(f"A descarregar Kaggle: {dataset_ref}...")
    
    pasta_destino = f"../data/raw_kaggle/{dataset_ref.split('/')[1]}"
    os.makedirs(pasta_destino, exist_ok=True)
    
    try:
        # Fazer download e extrair o zip
        kaggle.api.dataset_download_files(dataset_ref, path=pasta_destino, unzip=True)
        
        # Encontrar o ficheiro CSV extraído
        ficheiros_csv = glob.glob(f"{pasta_destino}/*.csv")
        if not ficheiros_csv:
            print(f"Nenhum CSV encontrado para {dataset_ref}.")
            return [], []
            
        df = pd.read_csv(ficheiros_csv[0]).head(3000) 
        
        textos_human = []
        textos_ia = []
        
        # Estratégia genérica: Procurar palavras-chave nas colunas para adivinhar a estrutura
        for col in df.columns:
            if 'text' in col.lower() or 'abstract' in col.lower() or 'content' in col.lower():
                col_texto = col
            if 'label' in col.lower() or 'generated' in col.lower() or 'source' in col.lower():
                col_label = col
                
        # Separar os dados
        for _, row in df.iterrows():
            texto = processar_texto(str(row[col_texto]))
            if len(texto) < 50: continue
                
            valor_label = str(row[col_label]).lower()
            if str(classe_humano).lower() in valor_label or valor_label == '0':
                textos_human.append(texto)
            elif str(classe_ia).lower() in valor_label or valor_label == '1':
                textos_ia.append(texto)
                
        return textos_human, textos_ia
        
    except Exception as e:
        print(f"❌ Erro no Kaggle {dataset_ref}: {e}")
        return [], []

if __name__ == "__main__":
    import numpy as np
    os.makedirs('../data', exist_ok=True)
    dados_por_modelo = {classe: [] for classe in CLASSES_ALVO}
    
    print("A iniciar o download (Hugging Face + Kaggle)...\n")

    # 1. Hugging Face
    print("-> A extrair do Hugging Face (HC3)...")
    try:
        ds_hc3 = load_dataset("Hello-SimpleAI/HC3", name="all", split="train[:3000]")
        df_hc3 = ds_hc3.to_pandas()
        for text in df_hc3['human_answers']: dados_por_modelo['human'].append(processar_texto(text))
        for text in df_hc3['chatgpt_answers']: dados_por_modelo['openai'].append(processar_texto(text))
    except Exception as e: print(f"Erro no HC3: {e}")

    print("-> A extrair do Hugging Face (AI Detection Pile)...")
    try:
        ds_pile = load_dataset("artem9k/ai-text-detection-pile", split="train[:8000]")
        for row in ds_pile:
            fonte, texto_limpo = str(row['source']).lower(), processar_texto(row['text'])
            if not texto_limpo or len(texto_limpo) < 50: continue
            if 'human' in fonte: dados_por_modelo['human'].append(texto_limpo)
            elif 'gpt' in fonte or 'openai' in fonte: dados_por_modelo['openai'].append(texto_limpo)
            elif 'llama' in fonte or 'meta' in fonte: dados_por_modelo['meta'].append(texto_limpo)
            elif 'palm' in fonte or 'google' in fonte or 'gemini' in fonte: dados_por_modelo['google'].append(texto_limpo)
            elif 'mistral' in fonte: dados_por_modelo['mistral'].append(texto_limpo)
    except Exception as e: print(f"Erro no Pile: {e}")

    # 2. Kaggle
    print("\n-> A extrair do Kaggle (LLM Detect AI vs Student)...")
    h_text, ia_text = extrair_kaggle("prajwaldongre/llm-detect-ai-generated-vs-student-generated-text", classe_humano='0', classe_ia='1')
    dados_por_modelo['human'].extend(h_text)
    dados_por_modelo['openai'].extend(ia_text)

    print("-> A extrair do Kaggle (GPT vs Human Abstracts)...")
    h_text2, ia_text2 = extrair_kaggle("heleneeriksen/gpt-vs-human-a-corpus-of-research-abstracts", classe_humano='human', classe_ia='ai')
    dados_por_modelo['human'].extend(h_text2)
    dados_por_modelo['openai'].extend(ia_text2)

    # 3. Resumo e Guardar
    print("\n--- Resumo dos Dados Extraídos ---")
    dataframes_finais = []
    
    for modelo, textos in dados_por_modelo.items():
        textos_unicos = list(set([t for t in textos if len(t) > 50])) # Remover duplicados e curtos
        
        if len(textos_unicos) > 0:
            df_modelo = pd.DataFrame({'text': textos_unicos, 'label': modelo})
            dataframes_finais.append(df_modelo)
            df_modelo.to_csv(f'../data/{modelo}.csv', index=False)
            print(f"✅ {modelo.upper()}: {len(textos_unicos)} frases -> Guardado em {modelo}.csv")
        else:
            print(f"⚠️ {modelo.upper()}: 0 frases encontradas.")

    if dataframes_finais:
        df_mestre = pd.concat(dataframes_finais, ignore_index=True)
        df_mestre = df_mestre.sample(frac=1, random_state=42).reset_index(drop=True)
        df_mestre.to_csv('../data/dataset.csv', index=False)
        print(f"\n🚀 Tudo pronto! O dataset foi guardado e baralhado com sucesso!")