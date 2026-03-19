import os

# ==========================================
# CONFIGURAÇÕES
# ==========================================
FICHEIRO_LISTA = 'lista.txt'                # A tua lista principal
FICHEIRO_FALHAS = 'termos_falhados.txt'     # A lista dos termos que a Wiki não encontra

# Podes adicionar aqui quantos ficheiros quiseres juntar à tua lista principal!
# Exemplo: ['nova_lista.txt', 'termos_extraidos_mega.txt']
LISTAS_A_JUNTAR = [] 

def carregar_termos(caminho_ficheiro):
    """Função auxiliar para ler um ficheiro e devolver uma lista limpa."""
    if not os.path.exists(caminho_ficheiro):
        return []
    with open(caminho_ficheiro, 'r', encoding='utf-8') as f:
        return [linha.strip() for linha in f if linha.strip()]

def executar_manutencao():
    print("🛠️ A INICIAR MANUTENÇÃO MASTER DA LISTA...\n")
    print("-" * 60)

    # ---------------------------------------------------------
    # 1. CARREGAR A LISTA PRINCIPAL
    # ---------------------------------------------------------
    termos_totais = carregar_termos(FICHEIRO_LISTA)
    print(f"📄 Lidos {len(termos_totais)} termos da '{FICHEIRO_LISTA}' original.")

    # ---------------------------------------------------------
    # 2. JUNTAR AS NOVAS LISTAS
    # ---------------------------------------------------------
    termos_adicionados = 0
    for ficheiro_extra in LISTAS_A_JUNTAR:
        if os.path.exists(ficheiro_extra):
            termos_novos = carregar_termos(ficheiro_extra)
            termos_totais.extend(termos_novos)
            termos_adicionados += len(termos_novos)
            print(f"➕ Adicionados {len(termos_novos)} termos vindos de '{ficheiro_extra}'.")
        else:
            print(f"⚠️ Aviso: O ficheiro '{ficheiro_extra}' não foi encontrado. A saltar...")

    total_bruto = len(termos_totais)
    print(f"📊 Total bruto antes da limpeza: {total_bruto} termos.")
    print("-" * 60)

    # ---------------------------------------------------------
    # 3. REMOVER DUPLICADOS
    # ---------------------------------------------------------
    # O dict.fromkeys remove os repetidos mas mantém a ordem de chegada
    termos_unicos = list(dict.fromkeys(termos_totais))
    duplicados_removidos = total_bruto - len(termos_unicos)
    print(f"✂️  Removidos {duplicados_removidos} termos duplicados.")

    # ---------------------------------------------------------
    # 4. REMOVER OS TERMOS QUE FALHAM NA WIKI
    # ---------------------------------------------------------
    falhas_removidas = 0
    termos_finais = []
    
    termos_falhados = set(carregar_termos(FICHEIRO_FALHAS))
    if termos_falhados:
        print(f"🚫 Lidos {len(termos_falhados)} termos problemáticos de '{FICHEIRO_FALHAS}'.")
        for termo in termos_unicos:
            if termo not in termos_falhados:
                termos_finais.append(termo)
            else:
                falhas_removidas += 1
        print(f"🗑️  Removidos {falhas_removidas} termos que já tinham falhado na Wiki.")
    else:
        print(f"✅ Nenhum termo falhado para remover (ou '{FICHEIRO_FALHAS}' não existe).")
        termos_finais = termos_unicos

    print("-" * 60)

    # ---------------------------------------------------------
    # 5. GUARDAR O RESULTADO FINAL
    # ---------------------------------------------------------
    with open(FICHEIRO_LISTA, 'w', encoding='utf-8') as f:
        for termo in termos_finais:
            f.write(termo + "\n")

    print(f"🎉 SUCESSO! A tua '{FICHEIRO_LISTA}' foi atualizada e está imaculada.")
    print(f"🏆 Total final pronto a processar: {len(termos_finais)} termos.")

if __name__ == "__main__":
    executar_manutencao()