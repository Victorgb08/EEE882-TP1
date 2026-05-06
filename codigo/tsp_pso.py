import random
import numpy as np
import math

# Parâmetros do Problema
MAX_EVALS = 100000
SWARM_SIZE = 40
W = 0.7
C1 = 1.5
C2 = 1.5

def ler_instancia_tsp(arquivo):
    """Lê arquivo TSP e retorna as coordenadas das cidades."""
    coordenadas = []
    with open(arquivo, 'r') as f:
        linhas = f.readlines()

    lendo_coords = False
    for linha in linhas:
        linha = linha.strip()
        if linha == "NODE_COORD_SECTION":
            lendo_coords = True
            continue
        if linha == "EOF":
            break
        if lendo_coords:
            partes = linha.split()
            if len(partes) >= 3:
                x = float(partes[1])
                y = float(partes[2])
                coordenadas.append((x, y))
    return coordenadas

def distancia_att(cidade1, cidade2):
    """Calcula distância pseudo-Euclidiana (ATT) entre duas cidades."""
    x1, y1 = cidade1
    x2, y2 = cidade2
    dx = x1 - x2
    dy = y1 - y2
    rij = math.sqrt((dx**2 + dy**2) / 10.0)
    tij = round(rij)
    if tij < rij:
        return tij + 1
    return tij

def calcular_matriz_distancias(coordenadas):
    """Calcula matriz de distâncias entre todas as cidades."""
    n = len(coordenadas)
    matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = distancia_att(coordenadas[i], coordenadas[j])
            matriz[i][j] = dist
            matriz[j][i] = dist
    return matriz

def avaliar_tsp(rota, matriz_distancias):
    """Avalia o custo total de uma rota (soma das distâncias)."""
    n = len(rota)
    custo = 0
    for i in range(n):
        cidade_atual = rota[i]
        proxima_cidade = rota[(i + 1) % n]
        custo += matriz_distancias[cidade_atual][proxima_cidade]
    return custo

def inicializar_particula(n_cidades):
    """Inicializa uma partícula com uma permutação aleatória."""
    posicao = random.sample(range(n_cidades), n_cidades)
    return {
        'posicao': posicao,
        'pbest_pos': list(posicao),
        'pbest_val': np.inf
    }

def inicializar_enxame(tamanho, n_cidades):
    """Inicializa o enxame de partículas."""
    return [inicializar_particula(n_cidades) for _ in range(tamanho)]

def crossover_pmx(pos1, pos2):
    """Partially Mapped Crossover (PMX) para combinar duas rotas."""
    n = len(pos1)
    inicio = random.randint(0, n - 2)
    fim = random.randint(inicio + 1, n - 1)

    filho = [-1] * n
    filho[inicio:fim + 1] = pos1[inicio:fim + 1]

    mapeamento = {}
    for i in range(inicio, fim + 1):
        mapeamento[pos2[i]] = pos1[i]

    for i in range(n):
        if inicio <= i <= fim:
            continue
        cidade = pos2[i]
        while cidade in mapeamento:
            cidade = mapeamento[cidade]
        filho[i] = cidade

    return filho

def atualizar_particula(particula, gbest_pos, W, C1, C2):
    """Atualiza a partícula usando operadores de crossover."""
    r1 = random.random()
    r2 = random.random()

    nova_posicao = list(particula['posicao'])

    if r1 < C1 / (C1 + C2):
        nova_posicao = crossover_pmx(nova_posicao, particula['pbest_pos'])

    if r2 < C2 / (C1 + C2):
        nova_posicao = crossover_pmx(nova_posicao, gbest_pos)

    return nova_posicao

def executar_pso_tsp(arquivo_tsp, W=0.7, C1=1.5, C2=1.5, silencioso=True):
    """Executa Particle Swarm Optimization para o problema TSP."""
    coordenadas = ler_instancia_tsp(arquivo_tsp)
    matriz_distancias = calcular_matriz_distancias(coordenadas)
    n_cidades = len(coordenadas)

    enxame = inicializar_enxame(SWARM_SIZE, n_cidades)
    avaliacoes = 0
    gbest_pos = None
    gbest_val = np.inf
    historico_convergencia = []

    while avaliacoes < MAX_EVALS:
        for particula in enxame:
            valor = avaliar_tsp(particula['posicao'], matriz_distancias)
            avaliacoes += 1

            if valor < particula['pbest_val']:
                particula['pbest_val'] = valor
                particula['pbest_pos'] = list(particula['posicao'])

            if valor < gbest_val:
                gbest_val = valor
                gbest_pos = list(particula['posicao'])

        for particula in enxame:
            particula['posicao'] = atualizar_particula(particula, gbest_pos, W, C1, C2)

        historico_convergencia.append(gbest_val)

        if not silencioso and avaliacoes % 10000 == 0:
            print(f"Avaliações: {avaliacoes}, Melhor: {gbest_val:.2f}")

    if not silencioso:
        print(f"Melhor distância encontrada: {gbest_val:.2f}")

    return gbest_val, historico_convergencia

def realizar_grid_search(repeticoes_por_teste=5):
    """Testa diferentes combinações de W, C1 e C2."""
    valores_W = [0.4, 0.7, 0.9]
    valores_C1 = [1.0, 1.5, 2.0]
    valores_C2 = [1.0, 1.5, 2.0]
    arquivo = "./codigo/att48.tsp"

    print(f"Iniciando Grid Search TSP PSO...")
    print("-" * 65)
    print(f"{'W':<6} | {'C1':<6} | {'C2':<6} | {'Média Distância':<18} | {'Melhor Encontrado'}")
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
                    resultado, historico = executar_pso_tsp(arquivo, W, C1, C2, silencioso=True)
                    resultados.append(resultado)
                    historicos.append(historico)

                media_resultados = np.mean(resultados)
                melhor_absoluto = np.min(resultados)

                print(f"{W:<6} | {C1:<6} | {C2:<6} | {media_resultados:<18.2f} | {melhor_absoluto:.2f}")

                if media_resultados < melhor_media_geral:
                    melhor_media_geral = media_resultados
                    melhor_configuracao = (W, C1, C2)
                    melhor_historico = historicos[np.argmin(resultados)]

    print("-" * 65)
    print(f"Melhor configuração encontrada: W = {melhor_configuracao[0]}, C1 = {melhor_configuracao[1]}, C2 = {melhor_configuracao[2]}")
    print(f"Com média de: {melhor_media_geral:.2f}")

    return melhor_configuracao, melhor_historico

def main():
    realizar_grid_search(repeticoes_por_teste=5)

if __name__ == "__main__":
    main()
