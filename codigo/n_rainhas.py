import random
import numpy as np
from itertools import combinations

# Parâmetros Iniciais
N = 8
POP_SIZE = 100
MAX_EVALS = 100000
PROB_CROSSOVER = 0.8
PROB_MUTATION = 0.1

def calcular_conflitos(individuo):
    """
    Calcula o número de cheques (conflitos) nas diagonais.
    Como é uma permutação, não há conflitos em linhas ou colunas.
    """
    conflitos = 0
    for i,j in combinations(range(N),2):
        # Se a diferença entre colunas for igual à diferença entre linhas, estão na mesma diagonal
        if abs(i - j) == abs(individuo[i] - individuo[j]):
            conflitos += 1
    return conflitos

def gerar_populacao_inicial(tamanho):
    populacao = []
    for _ in range(tamanho):
        individuo = np.random.permutation(range(1, N + 1)).tolist()
        populacao.append(individuo)
    return populacao

def selecao_torneio(populacao, aptidoes, k=3):
    """Seleciona o melhor indivíduo entre k sorteados aleatoriamente."""
    selecionados = random.sample(list(zip(populacao, aptidoes)), k)
    # Ordena pelo menor número de conflitos
    return min(selecionados,key=lambda x: x[1])[0]

def cruzamento_ox(pai1, pai2):
    """
    Order Crossover (OX) para garantir que o filho seja uma permutação válida.
    """
    if random.random() > PROB_CROSSOVER:
        return pai1, pai2

    def ox(p1, p2):
        inicio, fim = sorted(random.sample(range(N), 2))
        filho = [0] * N
        # Copia a faixa do pai 1
        filho[inicio:fim] = p1[inicio:fim]

        # Preenche o restante com os genes do pai 2, na ordem em que aparecem
        p2_filtrado = [gene for gene in p2 if gene not in filho]
        idx_p2 = 0
        for i in range(N):
            if filho[i] == 0:
                filho[i] = p2_filtrado[idx_p2]
                idx_p2 += 1
        return filho

    return ox(pai1, pai2), ox(pai2, pai1)

def mutacao_swap(individuo):
    """Troca duas posições do cromossomo aleatoriamente."""
    if random.random() < PROB_MUTATION:
        idx1, idx2 = random.sample(range(N), 2)
        individuo[idx1], individuo[idx2] = individuo[idx2], individuo[idx1]

def executar_ag():
    populacao = gerar_populacao_inicial(POP_SIZE)
    avaliacoes = 0
    geracao = 0

    while avaliacoes < MAX_EVALS:
        aptidoes = [calcular_conflitos(ind) for ind in populacao]
        avaliacoes += POP_SIZE

        # Verifica se encontrou a solução ótima (0 conflitos)
        if 0 in aptidoes:
            idx_solucao = aptidoes.index(0)
            print(f"Solução encontrada na geração {geracao} com {avaliacoes} avaliações!")
            print(f"Tabuleiro (Solução): {populacao[idx_solucao]}")
            return geracao, avaliacoes, populacao[idx_solucao]

        nova_populacao = []

        # Elitismo: mantém o melhor da geração atual
        melhor_idx = aptidoes.index(min(aptidoes))
        nova_populacao.append(populacao[melhor_idx])

        # Gera o restante da população
        while len(nova_populacao) < POP_SIZE:
            pai1 = selecao_torneio(populacao, aptidoes)
            pai2 = selecao_torneio(populacao, aptidoes)

            filho1, filho2 = cruzamento_ox(pai1, pai2)

            mutacao_swap(filho1)
            mutacao_swap(filho2)

            nova_populacao.extend([filho1, filho2])

        # Garante o tamanho exato da população (caso o elitismo deixe ímpar)
        populacao = nova_populacao[:POP_SIZE]
        geracao += 1

    print("Critério de parada atingido sem encontrar solução perfeita.")
    return geracao, avaliacoes, None

def executar_ag(silencioso=False):
    """Executa o AG uma única vez. Adicionado o parâmetro 'silencioso' para não poluir o terminal nas repetições."""
    populacao = gerar_populacao_inicial(POP_SIZE)
    avaliacoes = 0
    geracao = 0

    while avaliacoes < MAX_EVALS:
        aptidoes = [calcular_conflitos(ind) for ind in populacao]
        avaliacoes += POP_SIZE

        if 0 in aptidoes:
            idx_solucao = aptidoes.index(0)
            if not silencioso:
                print(f"Solução encontrada na geração {geracao} com {avaliacoes} avaliações!")
                print(f"Tabuleiro (Solução): {populacao[idx_solucao]}")
            return geracao, avaliacoes, populacao[idx_solucao]

        nova_populacao = []
        melhor_idx = np.argmin(aptidoes)
        nova_populacao.append(populacao[melhor_idx])

        while len(nova_populacao) < POP_SIZE:
            pai1 = selecao_torneio(populacao, aptidoes)
            pai2 = selecao_torneio(populacao, aptidoes)

            filho1, filho2 = cruzamento_ox(pai1, pai2)

            mutacao_swap(filho1)
            mutacao_swap(filho2)

            nova_populacao.extend([filho1, filho2])

        populacao = nova_populacao[:POP_SIZE]
        geracao += 1

    if not silencioso:
        print("Critério de parada atingido sem encontrar solução perfeita.")
    return geracao, avaliacoes, None

def executar_bateria_testes(num_repeticoes=30):
    """Executa o algoritmo múltiplas vezes para extrair a média exigida no relatório."""
    print(f"Iniciando bateria de {num_repeticoes} testes para as N-Rainhas...")
    geracoes_lista = []
    avaliacoes_lista = []
    sucessos = 0

    for i in range(num_repeticoes):
        geracoes, avaliacoes, solucao = executar_ag(silencioso=True)
        if solucao is not None:
            sucessos += 1
            geracoes_lista.append(geracoes)
            avaliacoes_lista.append(avaliacoes)

        # Progresso simples no terminal
        print(f"Execução {i+1}/{num_repeticoes} concluída.", end="\r")

    print("\n\n--- Resultados Estatísticos ---")
    print(f"Taxa de Sucesso: {(sucessos/num_repeticoes)*100:.2f}%")
    if sucessos > 0:
        print(f"Média de Gerações: {np.mean(geracoes_lista):.2f} (Desvio Padrão: {np.std(geracoes_lista):.2f})")
        print(f"Média de Avaliações: {np.mean(avaliacoes_lista):.2f} (Desvio Padrão: {np.std(avaliacoes_lista):.2f})")
        print(f"Mínimo de Gerações: {np.min(geracoes_lista)} | Máximo de Gerações: {np.max(geracoes_lista)}")

def main():
    executar_ag()
    executar_bateria_testes(num_repeticoes=30)

if __name__ == "__main__":
    main()
