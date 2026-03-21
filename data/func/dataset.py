import pandas as pd
import os

def juntar_baralhar_csvs():
    PASTA_FUNC = os.path.dirname(os.path.abspath(__file__))
    # Sobe um nível para a pasta AP (raiz) e entra em 'data'
    PASTA_DATA = os.path.abspath(os.path.join(PASTA_FUNC, '..'))
    # 1. Nomes dos ficheiros CSV que queres juntar
    ficheiros_para_juntar = [
        os.path.join(PASTA_DATA, "data", "google.csv"),
        os.path.join(PASTA_DATA, "data", "openai.csv"),
        os.path.join(PASTA_DATA, "data", "anthropic.csv"),
        os.path.join(PASTA_DATA, "data", "meta.csv"),
        os.path.join(PASTA_DATA, "data", "human.csv")
    ]

    # 2. Nome do ficheiro final
    ficheiro_final = os.path.join(PASTA_DATA, "data", "dataset.csv")

    # Queremos garantir que lemos pelo menos o Text e Label de cada um
    colunas_base = ['Text', 'Label']
    lista_dataframes = []

    print("A iniciar a junção dos ficheiros...")

    for ficheiro in ficheiros_para_juntar:
        try:
            # Ler o CSV com o separador ';'
            df = pd.read_csv(ficheiro, sep=';')
            
            # Verificar se o ficheiro tem as colunas 'Text' e 'Label'
            falta_coluna = [col for col in colunas_base if col not in df.columns]
            
            if falta_coluna:
                print(f"  -> AVISO: Faltam as colunas {falta_coluna} no ficheiro '{ficheiro}'. Ignorado.")
                continue
                
            # Guardar apenas Text e Label (vamos recriar o ID depois)
            df_filtrado = df[colunas_base]
            lista_dataframes.append(df_filtrado)
            
            print(f"  -> '{ficheiro}' lido com sucesso ({len(df_filtrado)} linhas).")
            
        except FileNotFoundError:
            print(f"  -> ERRO: O ficheiro '{ficheiro}' não foi encontrado.")
        except Exception as e:
            print(f"  -> ERRO ao ler '{ficheiro}': {e}")

    # 3. Juntar, Baralhar e Atribuir ID
    if lista_dataframes:
        # Juntar todos os dataframes num só
        dataset_completo = pd.concat(lista_dataframes, ignore_index=True)
        
        dataset_completo = dataset_completo.sample(frac=1, random_state=42).reset_index(drop=True)
        
        dataset_completo['ID'] = [f"D1-{i+1}" for i in range(len(dataset_completo))]
        
        # Reordenar as colunas para o formato final desejado: ID;Text;Label
        dataset_completo = dataset_completo[['ID', 'Text', 'Label']]
        
        # Guardar num novo CSV
        dataset_completo.to_csv(ficheiro_final, sep=';', index=False)
        
        print(f"\n✅ SUCESSO! Ficheiros combinados, baralhados e com novos IDs.")
        print(f"Ficheiro guardado como '{ficheiro_final}' com {len(dataset_completo)} linhas.")
    else:
        print("\n❌ Nenhum ficheiro válido foi processado.")

if __name__ == "__main__":
    juntar_baralhar_csvs()