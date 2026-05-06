import matplotlib.pyplot as plt
import numpy as np
from .tsp_de import executar_de_tsp

def plotar_convergencia_tsp_de(F, CR):
    arquivo = "./codigo/att48.tsp"
    print(f"Executando TSP DE (F={F}, CR={CR}) para gerar o gráfico...")

    melhor_valor, historico = executar_de_tsp(arquivo, F, CR, silencioso=True)

    plt.figure(figsize=(8, 5))
    plt.plot(historico, label=f'DE (F={F}, CR={CR})', color='darkorange', linewidth=2)
    plt.title('Curva de Convergência - TSP (Differential Evolution)')
    plt.xlabel('Gerações')
    plt.ylabel('Distância Total')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    caminho_arquivo = './relatorio/convergencia_tsp_de.png'
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo com sucesso em: {caminho_arquivo}\nMelhor distância: {melhor_valor:.2f}")

def main():
    melhor_F = 0.5
    melhor_CR = 0.9
    plotar_convergencia_tsp_de(melhor_F, melhor_CR)

if __name__ == "__main__":
    main()
