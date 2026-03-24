import pandas as pd
import os
import argparse


def build_test_dataset(exemplos_path, subm1_labels_path, output_path=None):
    """
    Junta dataset-exemplos.csv + dataset-subm1-labels.csv num único
    dataset de teste com labels, removendo duplicados.
    """
    # Carregar ambos os ficheiros
    df_exemplos = pd.read_csv(exemplos_path, sep=';')
    df_exemplos.columns = df_exemplos.columns.str.strip().str.lower()

    df_subm1 = pd.read_csv(subm1_labels_path, sep=';')
    df_subm1.columns = df_subm1.columns.str.strip().str.lower()

    # Normalizar nomes das colunas
    for df in [df_exemplos, df_subm1]:
        if 'labels' in df.columns:
            df.rename(columns={'labels': 'Label'}, inplace=True)

    # Garantir colunas consistentes
    cols = ['Id', 'Text', 'Label']
    df_exemplos = df_exemplos[cols]
    df_subm1 = df_subm1[cols]

    # Juntar e remover duplicados por ID
    df_test = pd.concat([df_exemplos, df_subm1], ignore_index=True)
    antes = len(df_test)
    df_test = df_test.drop_duplicates(subset='Id', keep='first')
    depois = len(df_test)

    # Info
    print(f'  dataset-exemplos:     {len(df_exemplos)} textos')
    print(f'  dataset-subm1-labels: {len(df_subm1)} textos')
    if antes != depois:
        print(f'  Duplicados removidos: {antes - depois}')
    print(f'  Dataset de teste:     {depois} textos')
    print(f'  Distribuição:')
    print(f'  {df_test["Label"].value_counts().to_string()}')

    # Guardar
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        df_test.to_csv(output_path, sep=';', index=False, encoding='utf-8')
        print(f'  ✅ Guardado em {output_path}')

    return df_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Constrói dataset de teste combinado')
    parser.add_argument('--exemplos', default='../dataset-samples.csv',
                        help='Caminho para dataset-samples.csv')
    parser.add_argument('--subm1', default='../dataset-subm1-labels.csv',
                        help='Caminho para dataset-subm1-labels.csv')
    parser.add_argument('--output', default='../dataset-test.csv',
                        help='Caminho de saída')
    args = parser.parse_args()

    print('A construir dataset de teste...')
    build_test_dataset(args.exemplos, args.subm1, args.output)