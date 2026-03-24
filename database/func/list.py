import os

# 1. Configurações de Caminhos

# Detecta a pasta onde o script está (AP/func/)
PASTA_FUNC = os.path.dirname(os.path.abspath(__file__))
# Sobe um nível para chegar à raiz de 'data' e entra em 'resources'
PASTA_DATA = os.path.abspath(os.path.join(PASTA_FUNC, '..'))
PASTA_RESOURCES = os.path.join(PASTA_DATA, 'resources')

# Ficheiros alvo dentro de /data/resources/
FICHEIRO_LISTA = os.path.join(PASTA_RESOURCES, 'list.txt')
FICHEIRO_FALHAS = os.path.join(PASTA_RESOURCES, 'fail.txt')

# Podes adicionar caminhos de ficheiros externos aqui para fundir na lista principal
LISTAS_A_JUNTAR = [] 

# 2. Funções de Manipulação de Listas

def carregar_termos(caminho_ficheiro):
    """Lê um ficheiro de texto e devolve uma lista de linhas limpas."""
    if not os.path.exists(caminho_ficheiro):
        return []
    with open(caminho_ficheiro, 'r', encoding='utf-8') as f:
        # Remove espaços em branco e ignora linhas vazias
        return [linha.strip() for linha in f if linha.strip()]

def executar_manutencao():
    print("🛠️  A INICIAR MANUTENÇÃO DA LISTA DE RECURSOS...\n")
    print(f"📂 Pasta de trabalho: {PASTA_RESOURCES}")
    print("-" * 60)

    # 1. Carregar a lista atual
    if not os.path.exists(FICHEIRO_LISTA):
        print(f"❌ Erro: Ficheiro principal '{FICHEIRO_LISTA}' não encontrado.")
        return

    termos_totais = carregar_termos(FICHEIRO_LISTA)
    print(f"📄 Lidos {len(termos_totais)} termos da lista atual.")

    # 2. Fundir com listas extra (se houver)
    for ficheiro_extra in LISTAS_A_JUNTAR:
        if os.path.exists(ficheiro_extra):
            novos = carregar_termos(ficheiro_extra)
            termos_totais.extend(novos)
            print(f"➕ Fundidos {len(novos)} termos de: {os.path.basename(ficheiro_extra)}")

    # 3. Remover Duplicados (Mantendo a ordem original)
    total_antes = len(termos_totais)
    termos_unicos = list(dict.fromkeys(termos_totais))
    print(f"✂️  Removidos {total_antes - len(termos_unicos)} duplicados.")

    # 4. Filtrar termos que já falharam (fail.txt)
    termos_falhados = set(carregar_termos(FICHEIRO_FALHAS))
    termos_finais = [t for t in termos_unicos if t not in termos_falhados]
    
    if termos_falhados:
        print(f"🚫 Removidos {len(termos_unicos) - len(termos_finais)} termos presentes em 'fail.txt'.")
    else:
        print("✅ Nenhum termo de falha encontrado para filtrar.")

    # 5. Guardar a lista limpa
    with open(FICHEIRO_LISTA, 'w', encoding='utf-8') as f:
        for termo in termos_finais:
            f.write(termo + "\n")

    print("-" * 60)
    print(f"🎉 SUCESSO! Ficheiro '{os.path.basename(FICHEIRO_LISTA)}' atualizado.")
    print(f"🏆 Total final: {len(termos_finais)} termos prontos para processar.")

if __name__ == "__main__":
    executar_manutencao()