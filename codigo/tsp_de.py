import random
import numpy as np
import math

# Parâmetros do Problema
MAX_EVALS = 100000
POP_SIZE = 50
F = 0.5
CR = 0.9

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

def gerar_populacao_inicial(tamanho, n_cidades):
    """Gera população inicial com permutações aleatórias."""
    return [random.sample(range(n_cidades), n_cidades) for _ in range(tamanho)]

def mutacao_diferencial_permutacao(alvo, r1, r2, r3, CR):
    """Mutação diferencial adaptada para permutações usando operador DEX."""
    n = len(alvo)
    trial = list(r1)

    for i in range(n):
        if random.random() < CR:
            if r2[i] != r3[i]:
                if r2[i] in trial and r3[i] in trial:
                    idx2 = trial.index(r2[i])
                    idx3 = trial.index(r3[i])
                    trial[idx2], trial[idx3] = trial[idx3], trial[idx2]
                elif r2[i] in trial:
                    idx = trial.index(r2[i])
                    trial[i], trial[idx] = trial[idx], trial[i]
                elif r3[i] in trial:
                    idx = trial.index(r3[i])
                    trial[i], trial[idx] = trial[idx], trial[i]

    return trial

def executar_de_tsp(arquivo_tsp, F=0.5, CR=0.9, silencioso=True):
    """Executa Differential Evolution para o problema TSP."""
    coordenadas = ler_instancia_tsp(arquivo_tsp)
    matriz_distancias = calcular_matriz_distancias(coordenadas)
    n_cidades = len(coordenadas)

    populacao = gerar_populacao_inicial(POP_SIZE, n_cidades)
    avaliacoes = 0
    melhor_objetivo_global = np.inf
    historico_convergencia = []

    while avaliacoes < MAX_EVALS:
        for i in range(POP_SIZE):
            valor_atual = avaliar_tsp(populacao[i], matriz_distancias)
            avaliacoes += 1

            indices = list(range(POP_SIZE))
            indices.remove(i)
            r1, r2, r3 = random.sample(indices, 3)

            trial = mutacao_diferencial_permutacao(populacao[i], populacao[r1], populacao[r2], populacao[r3], CR)

            valor_trial = avaliar_tsp(trial, matriz_distancias)
            avaliacoes += 1

            if valor_trial < valor_atual:
                populacao[i] = trial

            if valor_trial < melhor_objetivo_global:
                melhor_objetivo_global = valor_trial

        historico_convergencia.append(melhor_objetivo_global)

        if not silencioso and avaliacoes % 10000 == 0:
            print(f"Avaliações: {avaliacoes}, Melhor: {melhor_objetivo_global:.2f}")

    if not silencioso:
        print(f"Melhor distância encontrada: {melhor_objetivo_global:.2f}")

    return melhor_objetivo_global, historico_convergencia

def realizar_grid_search(repeticoes_por_teste=5):
    """Testa diferentes combinações de F e CR."""
    valores_F = [0.3, 0.5, 0.8]
    valores_CR = [0.5, 0.7, 0.9]
    arquivo = "./codigo/att48.tsp"

    print(f"Iniciando Grid Search TSP DE...")
    print("-" * 55)
    print(f"{'F':<6} | {'CR':<6} | {'Média Distância':<18} | {'Melhor Encontrado'}")
    print("-" * 55)

    melhor_configuracao = None
    melhor_media_geral = np.inf
    melhor_historico = []

    for F in valores_F:
        for CR in valores_CR:
            resultados = []
            historicos = []

            for _ in range(repeticoes_por_teste):
                resultado, historico = executar_de_tsp(arquivo, F, CR, silencioso=True)
                resultados.append(resultado)
                historicos.append(historico)

            media_resultados = np.mean(resultados)
            melhor_absoluto = np.min(resultados)

            print(f"{F:<6} | {CR:<6} | {media_resultados:<18.2f} | {melhor_absoluto:.2f}")

            if media_resultados < melhor_media_geral:
                melhor_media_geral = media_resultados
                melhor_configuracao = (F, CR)
                melhor_historico = historicos[np.argmin(resultados)]

    print("-" * 55)
    print(f"Melhor configuração encontrada: F = {melhor_configuracao[0]}, CR = {melhor_configuracao[1]}")
    print(f"Com média de: {melhor_media_geral:.2f}")

    return melhor_configuracao, melhor_historico

def main():
    realizar_grid_search(repeticoes_por_teste=5)

if __name__ == "__main__":
    main()
