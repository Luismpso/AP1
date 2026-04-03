import pandas as pd
import numpy as np
import os
import argparse

def build_test_dataset(csv_paths, output_path=None):
    """
    Junta múltiplos CSVs num único dataset de teste com labels,
    removendo duplicados por Id.
    """
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
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

def adjust_human_ratio(test_csv, output_path=None, max_human_pct=0.29, seed=42):
    """
    Reduz os textos Human no dataset de teste para < 30%.
    As restantes classes ficam intactas.
    """
    df = pd.read_csv(test_csv, sep=';', encoding='utf-8-sig')
    df.columns = df.columns.str.strip().str.lower()
    if 'labels' in df.columns:
        df.rename(columns={'labels': 'label'}, inplace=True)
    df.columns = [c.title() for c in df.columns]

    humanos = df[df['Label'] == 'Human']
    outros = df[df['Label'] != 'Human']

    # n_human / (n_human + n_outros) <= max_human_pct
    # n_human <= max_human_pct * (n_human + n_outros)
    # n_human <= max_human_pct * n_outros / (1 - max_human_pct)
    n_human = int(max_human_pct * len(outros) / (1 - max_human_pct))
    n_human = min(n_human, len(humanos))

    print(f'  Human antes: {len(humanos)}  ({len(humanos)/len(df)*100:.1f}%)')
    print(f'  Human depois: {n_human}  ({n_human/(n_human+len(outros))*100:.1f}%)')

    humanos = humanos.sample(n=n_human, random_state=seed)
    df_out = pd.concat([humanos, outros], ignore_index=True)
    df_out = df_out.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f'  Total: {len(df_out)} textos')
    print(df_out['Label'].value_counts().to_string())

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        df_out.to_csv(output_path, sep=';', index=False, encoding='utf-8')
        print(f'  ✅ Guardado em {output_path}')

    return df_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Constrói dataset de teste combinado')
    parser.add_argument('--exemplos', default='dataset-samples.csv')
    parser.add_argument('--subm1', default='dataset-subm1-labels.csv')
    parser.add_argument('--subm2', default='dataset-subm2-labels.csv')
    parser.add_argument('--subm3', default='dataset-subm3-labels.csv')
    parser.add_argument('--output', default='dataset-test.csv')
    args = parser.parse_args()

    print('A construir dataset de teste...')
    build_test_dataset([args.exemplos, args.subm1, args.subm2, args.subm3], args.output)