import ollama
import os
import pandas as pd
import anthropic

# ==========================================
# Configurações e constantes
# ==========================================
FICHEIRO_ENTRADA = r'..\data\raw\gpt-vs-human-a-corpus-of-research-abstracts\data_set.csv'
FICHEIRO_SAIDA = r'..\data\dataset.csv'

# Chave da API da Anthropic (Substitui pela tua chave completa)
CHAVE_ANTHROPIC = "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" 

# Quantidade de exemplos por classe
EXEMPLOS_POR_CLASSE = 493 

# Mapeamento dos modelos locais
MODELOS_OLLAMA = {
    'gemma2:latest': 'Google',  
    'llama3.1:latest': 'Meta' 
}

# ==========================================
# Funções auxiliares
# ==========================================
def contar(texto):
    if pd.isna(texto): return 0
    return len(str(texto).split())

def truncar(texto):
    if pd.isna(texto): return texto
    palavras = str(texto).split()
    if len(palavras) > 120:
        return " ".join(palavras[:120])
    return str(texto)

def guardar(lista_dados, label):
    """Guarda imediatamente os dados de uma classe específica num CSV de backup."""
    if not lista_dados: return
    
    pasta_saida = os.path.dirname(FICHEIRO_SAIDA)
    os.makedirs(pasta_saida, exist_ok=True) # Garante que a pasta existe
    
    caminho_csv = os.path.join(pasta_saida, f"{label}.csv")
    dados_filtrados = [d for d in lista_dados if d['Label'] == label]
    
    if dados_filtrados:
        df = pd.DataFrame(dados_filtrados)
        df.to_csv(caminho_csv, sep=';', index=False, encoding='utf-8')
        print(f"   [Salvo] -> {caminho_csv} ({len(dados_filtrados)} textos garantidos!)\n")

# ==========================================
# Passo 1: Extrair dados originais e os titulos
# ==========================================
def extrair(df):
    print(f"\n-> A procurar {EXEMPLOS_POR_CLASSE} pares (Human + OpenAI) com o mesmo título...")
    
    df_temp = df.copy()
    df_temp['num_palavras'] = df_temp['abstract'].apply(contar)
    
    df_validos = df_temp[df_temp['num_palavras'] >= 80].copy()
    df_validos['Label'] = df_validos['is_ai_generated'].map({0: 'Human', 1: 'OpenAI'})
    
    df_h = df_validos[df_validos['Label'] == 'Human'][['title', 'abstract', 'num_palavras']]
    df_o = df_validos[df_validos['Label'] == 'OpenAI'][['title', 'abstract', 'num_palavras']]
    
    pares = pd.merge(df_h, df_o, on='title', suffixes=('_H', '_O'))
    pares['mutuamente_perfeito'] = (pares['num_palavras_H'] <= 120) & (pares['num_palavras_O'] <= 120)
    
    pares_perfeitos = pares[pares['mutuamente_perfeito']]
    pares_imperfeitos = pares[~pares['mutuamente_perfeito']]
    
    if len(pares_perfeitos) >= EXEMPLOS_POR_CLASSE:
        pares_selecionados = pares_perfeitos.sample(n=EXEMPLOS_POR_CLASSE, random_state=42)
        qtd_mutuamente_perfeitos = EXEMPLOS_POR_CLASSE
    else:
        faltam = EXEMPLOS_POR_CLASSE - len(pares_perfeitos)
        pares_reserva = pares_imperfeitos.sample(n=faltam, random_state=42)
        pares_selecionados = pd.concat([pares_perfeitos, pares_reserva])
        qtd_mutuamente_perfeitos = len(pares_perfeitos)
        
    registos = []
    truncados = 0
    titulos_alvo = [] 
    
    for _, row in pares_selecionados.iterrows():
        titulos_alvo.append(row['title']) 
        
        txt_h = row['abstract_H']
        if row['num_palavras_H'] > 120:
            txt_h = truncar(txt_h)
            truncados += 1
        registos.append({'Text': txt_h, 'Label': 'Human'})
        
        txt_o = row['abstract_O']
        if row['num_palavras_O'] > 120:
            txt_o = truncar(txt_o)
            truncados += 1
        registos.append({'Text': txt_o, 'Label': 'OpenAI'})
        
    print(f"   [+] Extração concluída: {len(pares_selecionados)} pares extraídos ({len(registos)} textos).")
    
    # Faz o backup imediato!
    guardar(registos, 'Human')
    guardar(registos, 'OpenAI')
    
    return registos, truncados, qtd_mutuamente_perfeitos, titulos_alvo

# ==========================================
# Passo 2: Gerar novos dados com a Anthropic (Claude)
# ==========================================
def gerar_claude(titulos_alvo):
    print(f"\n-> A iniciar a geração com Anthropic (Claude) usando os {len(titulos_alvo)} títulos exatos...")
    dados_gerados = []
    truncados = 0
    perfeitos = 0
    gerados_claude = 0
    
    try:
        cliente_anthropic = anthropic.Anthropic(api_key=CHAVE_ANTHROPIC)
    except Exception as e:
        print(f"   [!] Erro ao inicializar o cliente Anthropic: {e}")
        return [], 0, 0

    for titulo in titulos_alvo:
        prompt = f"Write ONLY the pure text of a scientific abstract about the research title: '{titulo}'. The response MUST be strictly between 80 and 120 words. DO NOT include the title, DO NOT include the word 'Abstract', DO NOT include keywords, and DO NOT use any markdown formatting. Output absolutely nothing else but the sentences of the abstract."
        
        melhor_texto = ""
        max_palavras = 0
        original_era_maior = False
        
        for tentativa in range(3):
            try:
                resposta = cliente_anthropic.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=250,
                    messages=[{"role": "user", "content": prompt}]
                )
                texto_gerado = resposta.content[0].text.replace('\n', ' ').strip()
                num_palavras_original = contar(texto_gerado)
                texto_truncado = truncar(texto_gerado)
                num_palavras_final = contar(texto_truncado)
                
                # Guarda o recorde da tentativa mais longa
                if num_palavras_final > max_palavras:
                    melhor_texto = texto_truncado
                    max_palavras = num_palavras_final
                    original_era_maior = (num_palavras_original > 120)
                
                if num_palavras_final >= 80:
                    dados_gerados.append({'Text': texto_truncado, 'Label': 'Anthropic'})
                    gerados_claude += 1
                    
                    if num_palavras_original > 120: truncados += 1
                    else: perfeitos += 1
                        
                    print(f"   [+] Sucesso Anthropic ({gerados_claude}/{EXEMPLOS_POR_CLASSE}) - {num_palavras_final} pal.")
                    print(f"       Texto: \"{texto_truncado}\"\n")
                    break 
                else:
                    print(f"   [-] Tentativa {tentativa+1}/3 muito curta ({num_palavras_final} pal). A tentar de novo...")
            except Exception as e:
                print(f"   [!] Erro na API Anthropic: {e}")
                break 
        else:
            # Fallback se falhar as 3 vezes
            dados_gerados.append({'Text': melhor_texto, 'Label': 'Anthropic'})
            gerados_claude += 1
            if original_era_maior: truncados += 1
            else: perfeitos += 1
            print(f"   [!] Aceite por Fallback Anthropic ({gerados_claude}/{EXEMPLOS_POR_CLASSE}) - {max_palavras} pal.")
            print(f"       Texto: \"{melhor_texto}\"\n")
                
    # Faz o backup imediato da Anthropic!
    guardar(dados_gerados, 'Anthropic')
    
    return dados_gerados, truncados, perfeitos

# ==========================================
# Passo 3: Gerar novos dados com Ollama (Google e Meta)
# ==========================================
def gerar_ollama(titulos_alvo):
    print(f"\n-> A iniciar a geração com Ollama ({EXEMPLOS_POR_CLASSE} por modelo) usando os títulos exatos...")
    dados_gerados = []
    truncados = 0
    perfeitos = 0
    
    for modelo, label in MODELOS_OLLAMA.items():
        print(f"\n--- A gerar textos para a classe: {label} ({modelo}) ---")
        gerados_este_modelo = 0
        dados_deste_modelo = [] 
        
        for titulo in titulos_alvo:
            prompt = f"Write ONLY the pure text of a scientific abstract about the research title: '{titulo}'. The response MUST be strictly between 80 and 120 words. DO NOT include the title, DO NOT include the word 'Abstract', DO NOT include keywords, and DO NOT use any markdown formatting. Output absolutely nothing else but the sentences of the abstract."
            
            melhor_texto = ""
            max_palavras = 0
            original_era_maior = False
            
            for tentativa in range(3):
                try:
                    resposta = ollama.chat(model=modelo, messages=[{'role': 'user', 'content': prompt}])
                    texto_gerado = resposta['message']['content'].replace('\n', ' ').strip()
                    
                    num_palavras_original = contar(texto_gerado)
                    texto_truncado = truncar(texto_gerado)
                    num_palavras_final = contar(texto_truncado)
                    
                    # Guarda o recorde da tentativa mais longa
                    if num_palavras_final > max_palavras:
                        melhor_texto = texto_truncado
                        max_palavras = num_palavras_final
                        original_era_maior = (num_palavras_original > 120)
                    
                    if num_palavras_final >= 80:
                        registo = {'Text': texto_truncado, 'Label': label}
                        dados_gerados.append(registo)
                        dados_deste_modelo.append(registo)
                        gerados_este_modelo += 1
                        
                        if num_palavras_original > 120: truncados += 1
                        else: perfeitos += 1
                            
                        print(f"   [+] Sucesso {label} ({gerados_este_modelo}/{EXEMPLOS_POR_CLASSE}) - {num_palavras_final} pal.")
                        print(f"       Texto: \"{texto_truncado}\"\n")
                        break 
                    else:
                        print(f"   [-] Tentativa {tentativa+1}/3 muito curta ({num_palavras_final} pal). A tentar de novo...")
                except Exception as e:
                    print(f"   [!] Erro no modelo {modelo}: {e}")
                    break
            else:
                # Fallback se falhar as 3 vezes
                registo = {'Text': melhor_texto, 'Label': label}
                dados_gerados.append(registo)
                dados_deste_modelo.append(registo)
                gerados_este_modelo += 1
                if original_era_maior: truncados += 1
                else: perfeitos += 1
                print(f"   [!] Aceite por Fallback {label} ({gerados_este_modelo}/{EXEMPLOS_POR_CLASSE}) - {max_palavras} pal.")
                print(f"       Texto: \"{melhor_texto}\"\n")
        
        # Faz o backup imediato mal acaba este modelo (Google ou Meta)!
        guardar(dados_deste_modelo, label)
                
    return dados_gerados, truncados, perfeitos

# ==========================================
# Fluxo principal
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(FICHEIRO_ENTRADA):
        print(f"Erro: O ficheiro {FICHEIRO_ENTRADA} não foi encontrado.")
        exit()

    df_original = pd.read_csv(FICHEIRO_ENTRADA)
    
    # 1. Extrair os pares originais e os títulos 
    lista_originais, trunc_orig, qtd_pares_mutuamente_perfeitos, titulos_alvo = extrair(df_original)
    
    # 2. Gerar com Claude
    lista_claude, trunc_cla, perf_cla = gerar_claude(titulos_alvo)
    
    # 3. Gerar com Ollama
    lista_ollama, trunc_oll, perf_oll = gerar_ollama(titulos_alvo)

    # 4. Compilar métricas e guardar o dataset 
    total_truncados = trunc_orig + trunc_oll + trunc_cla
    
    dataset_final = lista_originais + lista_claude + lista_ollama
    df_final = pd.DataFrame(dataset_final)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    df_final.insert(0, 'ID', [f"D1-{i+1}" for i in range(len(df_final))])
    df_final.to_csv(FICHEIRO_SAIDA, sep=';', index=False, encoding='utf-8')
    
    print("\n==========================================")
    print(f"Dataset concluído e guardado como '{FICHEIRO_SAIDA}'!")
    print(f"Tamanho do dataset: {len(df_final)}")
    print("\nMétricas:")
    print(f" -> Pares originais extraídos: {qtd_pares_mutuamente_perfeitos*2} textos")
    print(f" -> Textos gerados: {perf_oll + perf_cla} textos")
    print(f" -> Total de textos truncados: {total_truncados}")
    print("\nDistribuição das classes:")
    print(df_final['Label'].value_counts())
    print("==========================================")