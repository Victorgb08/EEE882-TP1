import random
import numpy as np
import math

# Parâmetros do Problema
MAX_EVALS = 100000
POP_SIZE = 100

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

def selecao_torneio(populacao, aptidoes, k=3):
    """Seleciona o melhor indivíduo entre k sorteados aleatoriamente."""
    selecionados = random.sample(list(zip(populacao, aptidoes)), k)
    return min(selecionados, key=lambda x: x[1])[0]

def ordered_crossover(pai1, pai2):
    """Ordered Crossover (OX) para permutações."""
    n = len(pai1)
    inicio = random.randint(0, n - 2)
    fim = random.randint(inicio + 1, n - 1)

    filho = [-1] * n
    filho[inicio:fim + 1] = pai1[inicio:fim + 1]

    pos = (fim + 1) % n
    for cidade in pai2:
        if cidade not in filho:
            filho[pos] = cidade
            pos = (pos + 1) % n

    return filho

def mutacao_inversion(individuo, pm):
    """Mutação por inversão: inverte um segmento da rota."""
    if random.random() < pm:
        n = len(individuo)
        inicio = random.randint(0, n - 2)
        fim = random.randint(inicio + 1, n - 1)
        individuo[inicio:fim + 1] = reversed(individuo[inicio:fim + 1])

def busca_local_2opt(rota, matriz_distancias):
    """Aplica a heurística 2-opt para resolver cruzamentos na rota localmente."""
    n = len(rota)
    melhorou = True
    while melhorou:
        melhorou = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                n_i_prev, n_i = rota[i-1], rota[i]
                n_j_prev, n_j = rota[j-1], rota[j] if j < n - 1 else rota[0]
                
                d_atual = matriz_distancias[n_i_prev][n_i] + matriz_distancias[n_j_prev][n_j]
                d_novo = matriz_distancias[n_i_prev][n_j_prev] + matriz_distancias[n_i][n_j]
                
                if d_novo < d_atual:
                    rota[i:j] = rota[i:j][::-1]
                    melhorou = True
    return rota
# ---------------------------------------------------

def executar_ga_tsp(arquivo_tsp, pc=0.9, pm=0.05, silencioso=True):
    """Executa Algoritmo Genético (Memético) para o problema TSP."""
    coordenadas = ler_instancia_tsp(arquivo_tsp)
    matriz_distancias = calcular_matriz_distancias(coordenadas)
    n_cidades = len(coordenadas)

    populacao = gerar_populacao_inicial(POP_SIZE, n_cidades)
    avaliacoes = 0
    melhor_objetivo_global = np.inf
    historico_convergencia = []

    elitism_count = int(0.05 * POP_SIZE)

    while avaliacoes < MAX_EVALS:
        aptidoes = [avaliar_tsp(ind, matriz_distancias) for ind in populacao]
        avaliacoes += POP_SIZE

        # Identifica o melhor da geração e atualiza o histórico
        melhor_valor_geracao = min(aptidoes)
        if melhor_valor_geracao < melhor_objetivo_global:
            melhor_objetivo_global = melhor_valor_geracao

        historico_convergencia.append(melhor_objetivo_global)

        # Ordena a população para extrair a elite
        populacao_ordenada = [x for _, x in sorted(zip(aptidoes, populacao))]
        nova_populacao = [list(ind) for ind in populacao_ordenada[:elitism_count]]

        while len(nova_populacao) < POP_SIZE:
            pai1 = selecao_torneio(populacao, aptidoes)
            pai2 = selecao_torneio(populacao, aptidoes)

            if random.random() < pc:
                filho1 = ordered_crossover(pai1, pai2)
                filho2 = ordered_crossover(pai2, pai1)
            else:
                filho1 = list(pai1)
                filho2 = list(pai2)

            mutacao_inversion(filho1, pm)
            mutacao_inversion(filho2, pm)

            # Refina os filhos recém-nascidos antes de irem para a população
            filho1 = busca_local_2opt(filho1, matriz_distancias)
            filho2 = busca_local_2opt(filho2, matriz_distancias)
            # ---------------------------------------------------

            nova_populacao.extend([filho1, filho2])

        populacao = nova_populacao[:POP_SIZE]

    if not silencioso:
        print(f"Melhor distância encontrada: {melhor_objetivo_global:.2f}")

    return melhor_objetivo_global, historico_convergencia

def realizar_grid_search(repeticoes_por_teste=5):
    """Testa diferentes combinações de pc e pm."""
    valores_pc = [0.7, 0.9]
    valores_pm = [0.01, 0.05]
    arquivo = "./codigo/att48.tsp"

    print(f"Iniciando Grid Search TSP Memético (Avaliando {len(valores_pc) * len(valores_pm)} combinações, {repeticoes_por_teste} vezes cada)...")
    print("-" * 55)
    print(f"{'p_c':<6} | {'p_m':<6} | {'Média Distância':<18} | {'Melhor Encontrado'}")
    print("-" * 55)

    melhor_configuracao = None
    melhor_media_geral = np.inf

    for pc in valores_pc:
        for pm in valores_pm:
            resultados = []

            for _ in range(repeticoes_por_teste):
                resultado, _ = executar_ga_tsp(arquivo, pc, pm, silencioso=True)
                resultados.append(resultado)

            media_resultados = np.mean(resultados)
            melhor_absoluto = np.min(resultados)

            print(f"{pc:<6} | {pm:<6} | {media_resultados:<18.2f} | {melhor_absoluto:.2f}")

            if media_resultados < melhor_media_geral:
                melhor_media_geral = media_resultados
                melhor_configuracao = (pc, pm)

    print("-" * 55)
    print(f"Melhor configuração encontrada: p_c = {melhor_configuracao[0]}, p_m = {melhor_configuracao[1]}")
    print(f"Com média de: {melhor_media_geral:.2f} (Ótimo conhecido: 10628)")

    return melhor_configuracao

if __name__ == "__main__":
    realizar_grid_search(repeticoes_por_teste=3)