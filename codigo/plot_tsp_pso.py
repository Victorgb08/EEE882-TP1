import matplotlib.pyplot as plt
import numpy as np
from .tsp_pso import executar_pso_tsp

def plotar_convergencia_tsp_pso(W, C1, C2):
    arquivo = "./codigo/att48.tsp"
    print(f"Executando TSP PSO (W={W}, C1={C1}, C2={C2}) para gerar o gráfico...")

    melhor_valor, historico = executar_pso_tsp(arquivo, W, C1, C2, silencioso=True)

    plt.figure(figsize=(8, 5))
    plt.plot(historico, label=f'PSO (W={W}, C1={C1}, C2={C2})', color='green', linewidth=2)
    plt.title('Curva de Convergência - TSP (Particle Swarm Optimization)')
    plt.xlabel('Iterações')
    plt.ylabel('Distância Total')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    caminho_arquivo = './relatorio/convergencia_tsp_pso.png'
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo com sucesso em: {caminho_arquivo}\nMelhor distância: {melhor_valor:.2f}")

def main():
    melhor_W = 0.7
    melhor_C1 = 1.5
    melhor_C2 = 1.5
    plotar_convergencia_tsp_pso(melhor_W, melhor_C1, melhor_C2)

if __name__ == "__main__":
    main()
