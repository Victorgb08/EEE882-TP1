import random
import numpy as np

# Parâmetros do Problema
N_DIM = 10
BOUNDS = [-5.12, 5.12]
MAX_EVALS = 100000

# Hiperparâmetros PSO
SWARM_SIZE = 40
W = 0.7
C1 = 1.5
C2 = 1.5

def funcao_rastrigin(x):
    """Avalia a função Rastrigin para n dimensões."""
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def inicializar_particula():
    """Inicializa uma partícula com posição, velocidade e melhores posições."""
    posicao = np.array([random.uniform(BOUNDS[0], BOUNDS[1]) for _ in range(N_DIM)])
    velocidade = np.array([random.uniform(-1, 1) for _ in range(N_DIM)])
    return {
        'posicao': posicao,
        'velocidade': velocidade,
        'pbest_pos': np.copy(posicao),
        'pbest_val': np.inf
    }

def inicializar_enxame(tamanho):
    """Inicializa o enxame de partículas."""
    return [inicializar_particula() for _ in range(tamanho)]

def atualizar_velocidade(particula, gbest_pos, W, C1, C2):
    """Atualiza a velocidade da partícula."""
    r1 = random.random()
    r2 = random.random()
    inercia = W * particula['velocidade']
    cognitivo = C1 * r1 * (particula['pbest_pos'] - particula['posicao'])
    social = C2 * r2 * (gbest_pos - particula['posicao'])
    return inercia + cognitivo + social

def limitar_velocidade(velocidade, v_max=2.0):
    """Limita a velocidade máxima."""
    return np.clip(velocidade, -v_max, v_max)

def executar_pso_rastrigin(W=0.7, C1=1.5, C2=1.5, silencioso=True):
    """Executa Particle Swarm Optimization para a Função Rastrigin."""
    enxame = inicializar_enxame(SWARM_SIZE)
    avaliacoes = 0
    gbest_pos = None
    gbest_val = np.inf
    historico_convergencia = []

    while avaliacoes < MAX_EVALS:
        for particula in enxame:
            # Avalia a partícula
            valor = funcao_rastrigin(particula['posicao'])
            avaliacoes += 1

            # Atualiza pbest
            if valor < particula['pbest_val']:
                particula['pbest_val'] = valor
                particula['pbest_pos'] = np.copy(particula['posicao'])

            # Atualiza gbest
            if valor < gbest_val:
                gbest_val = valor
                gbest_pos = np.copy(particula['posicao'])

        # Atualiza posições e velocidades
        for particula in enxame:
            particula['velocidade'] = atualizar_velocidade(particula, gbest_pos, W, C1, C2)
            particula['velocidade'] = limitar_velocidade(particula['velocidade'])
            particula['posicao'] = particula['posicao'] + particula['velocidade']
            particula['posicao'] = np.clip(particula['posicao'], BOUNDS[0], BOUNDS[1])

        historico_convergencia.append(gbest_val)

        if not silencioso and avaliacoes % 10000 == 0:
            print(f"Avaliações: {avaliacoes}, Melhor: {gbest_val:.6f}")

    if not silencioso:
        print(f"Melhor valor Rastrigin encontrado: {gbest_val:.6f}")

    return gbest_val, historico_convergencia

def realizar_grid_search(repeticoes_por_teste=5):
    """Testa diferentes combinações de W, C1 e C2 para encontrar a melhor configuração."""
    valores_W = [0.4, 0.7, 0.9]
    valores_C1 = [1.0, 1.5, 2.0]
    valores_C2 = [1.0, 1.5, 2.0]

    print(f"Iniciando Grid Search PSO...")
    print("-" * 65)
    print(f"{'W':<6} | {'C1':<6} | {'C2':<6} | {'Média Rastrigin':<18} | {'Melhor Encontrado'}")
    print("-" * 65)

    melhor_configuracao = None
    melhor_media_geral = np.inf
    melhor_historico = []

    for W in valores_W:
        for C1 in valores_C1:
            for C2 in valores_C2:
                resultados = []
                historicos = []

                for _ in range(repeticoes_por_teste):
                    resultado, historico = executar_pso_rastrigin(W, C1, C2, silencioso=True)
                    resultados.append(resultado)
                    historicos.append(historico)

                media_resultados = np.mean(resultados)
                melhor_absoluto = np.min(resultados)

                print(f"{W:<6} | {C1:<6} | {C2:<6} | {media_resultados:<18.6f} | {melhor_absoluto:.6f}")

                if media_resultados < melhor_media_geral:
                    melhor_media_geral = media_resultados
                    melhor_configuracao = (W, C1, C2)
                    melhor_historico = historicos[np.argmin(resultados)]

    print("-" * 65)
    print(f"Melhor configuração encontrada: W = {melhor_configuracao[0]}, C1 = {melhor_configuracao[1]}, C2 = {melhor_configuracao[2]}")
    print(f"Com média de: {melhor_media_geral:.6f}")

    return melhor_configuracao, melhor_historico

def main():
    realizar_grid_search(repeticoes_por_teste=5)

if __name__ == "__main__":
    main()
