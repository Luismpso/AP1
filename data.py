import pandas as pd
import ollama
import time
import os

def gerar_frases_ollama(modelo_ollama, label_ia, quantidade):
    """Gera frases curtas usando o Ollama local."""
    textos_gerados = []
    print(f"\n⏳ A gerar {quantidade} frases para a classe '{label_ia}' usando o modelo '{modelo_ollama}'...")
    
    prompt = """
    Gera apenas UMA frase em inglês sobre um tema aleatório (ciência, história, desporto, tecnologia, culinária, etc).
    A frase DEVE ter obrigatoriamente entre 80 e 110 caracteres no total.
    Não uses aspas, não faças listas, não dês explicações. Escreve apenas a frase e nada mais.
    """
    
    for i in range(quantidade):
        try:
            response = ollama.chat(model=modelo_ollama, messages=[{'role': 'user', 'content': prompt}])
            texto = response['message']['content'].strip()
            # Limpar aspas caso o modelo as coloque por teimosia
            texto = texto.replace('"', '').replace("'", "")
            textos_gerados.append(texto)
            print(f"[{i+1}/{quantidade}] {texto}")
        except Exception as e:
            print(f"⚠️ Erro ao gerar: {e}. A tentar novamente em 2 segundos...")
            time.sleep(2)
            
    return pd.DataFrame({'text': textos_gerados, 'label': label_ia})

if __name__ == "__main__":
    caminho_dataset = '../data/dataset.csv'
    
    if not os.path.exists(caminho_dataset):
        print(f"❌ Erro: Não encontrei o {caminho_dataset}. Corre o download_data.py primeiro!")
        exit()

    # 1. Carregar o dataset atual
    df = pd.read_csv(caminho_dataset).dropna()
    contagens = df['label'].value_counts()
    
    print("📊 Contagem:")
    print(contagens)
    
    # 2. Definir o alvo (Quantidade de textos da classe 'human')
    if 'human' in contagens:
        alvo = contagens['human']
    else:
        alvo = int(input("Não encontrei a classe 'human'. Introduz o número de frases alvo para todas as classes: "))
        
    print(f"\n🎯 OBJETIVO: Deixar TODAS as classes exatamente com {alvo} frases.")
    
    # Mapeamento: Qual modelo do Ollama usar para cada label?
    modelos_ollama = {
        'meta': 'llama3',  
        'mistral': 'mistral',
        'google': 'gemma',  
        'openai': 'llama3' 
    }
    
    dataframes_finais = []
    
    # 3. Processar cada classe para equilibrar
    for label in ['human', 'openai', 'meta', 'google', 'mistral']:
        df_classe = df[df['label'] == label]
        qtd_atual = len(df_classe)
        
        if qtd_atual == alvo:
            print(f"✅ {label.upper()}: Já está perfeito ({alvo} frases).")
            dataframes_finais.append(df_classe)
            
        elif qtd_atual > alvo:
            print(f"✂️ {label.upper()}: Tem {qtd_atual} frases. A cortar para {alvo} (Undersampling)...")
            df_cortado = df_classe.sample(n=alvo, random_state=42)
            dataframes_finais.append(df_cortado)
            
        elif qtd_atual < alvo:
            falta = alvo - qtd_atual
            print(f"🤖 {label.upper()}: Tem apenas {qtd_atual} frases. Faltam {falta}!")
            
            if label == 'human':
                print("⚠️ AVISO: Faltam humanos! Não podemos gerar humanos com IA. Tenta sacar mais dados reais.")
                dataframes_finais.append(df_classe)
            else:
                # Gerar o que falta com o Ollama
                modelo = modelos_ollama.get(label, 'llama3')
                df_gerado = gerar_frases_ollama(modelo_ollama=modelo, label_ia=label, quantidade=falta)
                
                # Juntar os dados originais da classe com os gerados
                df_completo_classe = pd.concat([df_classe, df_gerado])
                dataframes_finais.append(df_completo_classe)

    # 4. Juntar tudo, baralhar e salvar
    df_final_equilibrado = pd.concat(dataframes_finais, ignore_index=True)
    df_final_equilibrado = df_final_equilibrado.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Substituir o dataset antigo pelo novo (perfeito!)
    df_final_equilibrado.to_csv(caminho_dataset, index=False)
    
    print("\n🎉 Sucesso Absuluto!")
    print("Nova contagem do dataset equilibrado:")
    print(df_final_equilibrado['label'].value_counts())
    print(f"Dataset guardado e pronto para treinar em: {caminho_dataset}")