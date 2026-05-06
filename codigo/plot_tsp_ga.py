import matplotlib.pyplot as plt
import numpy as np
from .tsp_ga import executar_ga_tsp

def plotar_convergencia_tsp_ga(pc, pm):
    arquivo = "./codigo/att48.tsp"
    print(f"Executando TSP GA (pc={pc}, pm={pm}) para gerar o gráfico...")

    melhor_valor, historico = executar_ga_tsp(arquivo, pc, pm, silencioso=True)
    avaliacoes = np.arange(len(historico)) * 100

    plt.figure(figsize=(8, 5))
    plt.plot(avaliacoes, historico, label=f'GA (pc={pc}, pm={pm})', color='royalblue', linewidth=2)
    plt.title('Curva de Convergência - TSP (Genetic Algorithm)')
    plt.xlabel('Avaliações')
    plt.ylabel('Distância Total')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    caminho_arquivo = './relatorio/convergencia_tsp_ga.png'
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo com sucesso em: {caminho_arquivo}\nMelhor distância: {melhor_valor:.2f}")

def main():
    melhor_pc = 0.9
    melhor_pm = 0.05
    plotar_convergencia_tsp_ga(melhor_pc, melhor_pm)

if __name__ == "__main__":
    main()
