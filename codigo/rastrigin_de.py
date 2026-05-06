import random
import numpy as np

# Parâmetros do Problema
N_DIM = 10
BOUNDS = [-5.12, 5.12]
MAX_EVALS = 100000

# Hiperparâmetros DE
POP_SIZE = 50
F = 0.8
CR = 0.9

def funcao_rastrigin(x):
    """Avalia a função Rastrigin para n dimensões."""
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def gerar_populacao_inicial(tamanho):
    """Gera população inicial no espaço contínuo."""
    populacao = []
    for _ in range(tamanho):
        individuo = [random.uniform(BOUNDS[0], BOUNDS[1]) for _ in range(N_DIM)]
        populacao.append(individuo)
    return np.array(populacao)

def mutacao_de(populacao, i, F):
    """Mutação diferencial: v = x_r1 + F * (x_r2 - x_r3)"""
    indices = list(range(len(populacao)))
    indices.remove(i)
    r1, r2, r3 = random.sample(indices, 3)
    return populacao[r1] + F * (populacao[r2] - populacao[r3])

def cruzamento_de(alvo, mutante, CR):
    """Cruzamento binomial: troca componentes entre alvo e mutante."""
    trial = np.copy(alvo)
    j_rand = random.randint(0, N_DIM - 1)
    for j in range(N_DIM):
        if random.random() < CR or j == j_rand:
            trial[j] = mutante[j]
    return trial

def clip_limites(individuo):
    """Garante que o indivíduo esteja dentro dos limites."""
    return np.clip(individuo, BOUNDS[0], BOUNDS[1])

def executar_de_rastrigin(F=0.8, CR=0.9, silencioso=True):
    """Executa Differential Evolution para a Função Rastrigin."""
    populacao = gerar_populacao_inicial(POP_SIZE)
    avaliacoes = 0
    historico_convergencia = []
    
    #Avaliamos a população inicial uma única vez antes do loop
    custos_populacao = []
    for individuo in populacao:
        custos_populacao.append(funcao_rastrigin(individuo))
        avaliacoes += 1
        
    melhor_objetivo_global = min(custos_populacao)

    while avaliacoes < MAX_EVALS:
        for i in range(POP_SIZE):
            # O valor_atual agora é resgatado da lista, sem gastar avaliação
            valor_atual = custos_populacao[i]

            # Mutação
            mutante = mutacao_de(populacao, i, F)
            mutante = clip_limites(mutante)

            # Cruzamento
            trial = cruzamento_de(populacao[i], mutante, CR)

            # Seleção
            valor_trial = funcao_rastrigin(trial)
            avaliacoes += 1

            if valor_trial < valor_atual:
                populacao[i] = trial
                # Atualiza o custo do indivíduo na lista
                custos_populacao[i] = valor_trial

            if valor_trial < melhor_objetivo_global:
                melhor_objetivo_global = valor_trial

        historico_convergencia.append(melhor_objetivo_global)

        if not silencioso and avaliacoes % 10000 == 0:
            print(f"Avaliações: {avaliacoes}, Melhor: {melhor_objetivo_global:.6f}")

    if not silencioso:
        print(f"Melhor valor Rastrigin encontrado: {melhor_objetivo_global:.6f}")

    return melhor_objetivo_global, historico_convergencia

def realizar_grid_search(repeticoes_por_teste=5):
    """Testa diferentes combinações de F e CR para encontrar a melhor configuração."""
    valores_F = [0.5, 0.8, 1.0]
    valores_CR = [0.5, 0.7, 0.9]

    print(f"Iniciando Grid Search DE (Avaliando {len(valores_F) * len(valores_CR)} combinações, {repeticoes_por_teste} vezes cada)...")
    print("-" * 55)
    print(f"{'F':<6} | {'CR':<6} | {'Média Rastrigin':<18} | {'Melhor Encontrado'}")
    print("-" * 55)

    melhor_configuracao = None
    melhor_media_geral = np.inf
    melhor_historico = []

    for F in valores_F:
        for CR in valores_CR:
            resultados = []
            historicos = []

            for _ in range(repeticoes_por_teste):
                resultado, historico = executar_de_rastrigin(F, CR, silencioso=True)
                resultados.append(resultado)
                historicos.append(historico)

            media_resultados = np.mean(resultados)
            melhor_absoluto = np.min(resultados)

            print(f"{F:<6} | {CR:<6} | {media_resultados:<18.6f} | {melhor_absoluto:.6f}")

            if media_resultados < melhor_media_geral:
                melhor_media_geral = media_resultados
                melhor_configuracao = (F, CR)
                melhor_historico = historicos[np.argmin(resultados)]

    print("-" * 55)
    print(f"Melhor configuração encontrada: F = {melhor_configuracao[0]}, CR = {melhor_configuracao[1]}")
    print(f"Com média de: {melhor_media_geral:.6f}")

    return melhor_configuracao, melhor_historico

def main():
    realizar_grid_search(repeticoes_por_teste=5)

if __name__ == "__main__":
    main()