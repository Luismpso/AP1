import pandas as pd
import os
import argparse

def build_test_dataset(csv_paths, output_path=None):
    """
    Junta múltiplos CSVs num único dataset de teste com labels,
    removendo duplicados por Id.
    """
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path, sep=';')
        df.columns = df.columns.str.strip().str.lower()
        if 'labels' in df.columns:
            df.rename(columns={'labels': 'label'}, inplace=True)
        # Normalizar para Title Case
        df.columns = [c.title() for c in df.columns]
        df = df[['Id', 'Text', 'Label']]
        print(f'  {os.path.basename(path)}: {len(df)} textos')
        dfs.append(df)

    df_test = pd.concat(dfs, ignore_index=True)
    antes = len(df_test)
    df_test = df_test.drop_duplicates(subset='Id', keep='first')
    depois = len(df_test)

    if antes != depois:
        print(f'  Duplicados removidos: {antes - depois}')
    print(f'  Dataset de teste:     {depois} textos')
    print(f'  Distribuição:')
    print(f'  {df_test["Label"].value_counts().to_string()}')

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        df_test.to_csv(output_path, sep=';', index=False, encoding='utf-8')
        print(f'  ✅ Guardado em {output_path}')

    return df_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Constrói dataset de teste combinado')
    parser.add_argument('--exemplos', default='dataset-samples.csv')
    parser.add_argument('--subm1', default='dataset-subm1-labels.csv')
    parser.add_argument('--subm2', default='dataset-subm2-labels.csv')
    parser.add_argument('--output', default='dataset-test.csv')
    args = parser.parse_args()

    print('A construir dataset de teste...')
    build_test_dataset([args.exemplos, args.subm1, args.subm2], args.output)