import matplotlib.pyplot as plt
import numpy as np
from rastrigin import executar_ag_rastrigin

def plotar_convergencia_rastrigin(pc, pm):
    print(f"Executando Rastrigin (pc={pc}, pm={pm}) para gerar o gráfico...")
    
    # Executa uma vez com os melhores parâmetros para pegar o histórico
    melhor_valor, historico = executar_ag_rastrigin(pc, pm, silencioso=True)
    geracoes = np.arange(len(historico))

    # Configuração do Gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(geracoes, historico, label=f'Melhor Aptidão (pc={pc}, pm={pm})', color='royalblue', linewidth=2)
    plt.title('Curva de Convergência - Função Rastrigin')
    plt.xlabel('Gerações')
    plt.ylabel('Valor da Função Objetivo')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Salva diretamente na pasta do relatório
    caminho_arquivo = '../relatorio/convergencia_rastrigin.png'
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo com sucesso em: {caminho_arquivo}\nMelhor valor final: {melhor_valor:.6f}")

if __name__ == "__main__":
    # IMPORTANTE: Substitua estes valores pelos melhores que você encontrou no seu Grid Search!
    melhor_pc = 0.9 
    melhor_pm = 0.05
    plotar_convergencia_rastrigin(melhor_pc, melhor_pm)
