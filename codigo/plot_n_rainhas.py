import matplotlib.pyplot as plt
import numpy as np
from n_rainhas import executar_ag

def plotar_estatisticas_rainhas(num_repeticoes=30):
    print(f"Executando {num_repeticoes} testes das N-Rainhas para gerar o gráfico...")
    geracoes_lista = []

    for i in range(num_repeticoes):
        geracoes, avaliacoes, solucao = executar_ag(silencioso=True)
        if solucao is not None:
            geracoes_lista.append(geracoes)
        print(f"Progresso: {i+1}/{num_repeticoes}", end="\r")

    # Configuração do Gráfico
    plt.figure(figsize=(8, 5))
    plt.hist(geracoes_lista, bins=10, color='seagreen', alpha=0.7, edgecolor='black')
    plt.title('Distribuição de Gerações para Solução - 8-Rainhas')
    plt.xlabel('Número de Gerações até Convergir')
    plt.ylabel('Frequência (Nº de Execuções)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Salva diretamente na pasta do relatório
    caminho_arquivo = '../relatorio/histograma_rainhas.png'
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    print(f"\nGráfico salvo com sucesso em: {caminho_arquivo}")
    
    # Opcional: plt.show() para abrir a janela caso esteja usando interface gráfica no WSL

if __name__ == "__main__":
    plotar_estatisticas_rainhas()
