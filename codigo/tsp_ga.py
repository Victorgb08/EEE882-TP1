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

def selecao_roleta(populacao, aptidoes):
    """Seleciona um indivíduo proporcionalmente à sua aptidão (invertida)."""
    max_aptidao = max(aptidoes)
    aptidoes_inv = [max_aptidao - a + 1 for a in aptidoes]
    soma = sum(aptidoes_inv)
    probs = [a / soma for a in aptidoes_inv]
    return populacao[np.random.choice(len(populacao), p=probs)]

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

def pmx_crossover(pai1, pai2):
    """Partially Mapped Crossover (PMX) para permutações."""
    n = len(pai1)
    inicio = random.randint(0, n - 2)
    fim = random.randint(inicio + 1, n - 1)

    filho = [-1] * n
    filho[inicio:fim + 1] = pai1[inicio:fim + 1]

    mapeamento = {}
    for i in range(inicio, fim + 1):
        mapeamento[pai2[i]] = pai1[i]

    for i in range(n):
        if inicio <= i <= fim:
            continue
        cidade = pai2[i]
        while cidade in mapeamento:
            cidade = mapeamento[cidade]
        filho[i] = cidade

    return filho

def mutacao_swap(individuo, pm):
    """Mutação por troca: troca duas cidades aleatórias."""
    if random.random() < pm:
        n = len(individuo)
        i, j = random.sample(range(n), 2)
        individuo[i], individuo[j] = individuo[j], individuo[i]

def mutacao_inversion(individuo, pm):
    """Mutação por inversão: inverte um segmento da rota."""
    if random.random() < pm:
        n = len(individuo)
        inicio = random.randint(0, n - 2)
        fim = random.randint(inicio + 1, n - 1)
        individuo[inicio:fim + 1] = reversed(individuo[inicio:fim + 1])

def executar_ga_tsp(arquivo_tsp, pc=0.9, pm=0.05, crossover='ox', silencioso=True):
    """Executa Algoritmo Genético para o problema TSP."""
    coordenadas = ler_instancia_tsp(arquivo_tsp)
    matriz_distancias = calcular_matriz_distancias(coordenadas)
    n_cidades = len(coordenadas)

    populacao = gerar_populacao_inicial(POP_SIZE, n_cidades)
    avaliacoes = 0
    melhor_objetivo_global = np.inf
    historico_convergencia = []

    while avaliacoes < MAX_EVALS:
        aptidoes = [avaliar_tsp(ind, matriz_distancias) for ind in populacao]
        avaliacoes += POP_SIZE

        melhor_valor_geracao = min(aptidoes)
        if melhor_valor_geracao < melhor_objetivo_global:
            melhor_objetivo_global = melhor_valor_geracao

        historico_convergencia.append(melhor_objetivo_global)

        nova_populacao = []

        melhor_idx = np.argmin(aptidoes)
        nova_populacao.append(populacao[melhor_idx])

        while len(nova_populacao) < POP_SIZE:
            pai1 = selecao_torneio(populacao, aptidoes)
            pai2 = selecao_torneio(populacao, aptidoes)

            if random.random() < pc:
                if crossover == 'ox':
                    filho1 = ordered_crossover(pai1, pai2)
                    filho2 = ordered_crossover(pai2, pai1)
                else:
                    filho1 = pmx_crossover(pai1, pai2)
                    filho2 = pmx_crossover(pai2, pai1)
            else:
                filho1 = list(pai1)
                filho2 = list(pai2)

            mutacao_inversion(filho1, pm)
            mutacao_inversion(filho2, pm)

            nova_populacao.extend([filho1, filho2])

        populacao = nova_populacao[:POP_SIZE]

    if not silencioso:
        print(f"Melhor distância encontrada: {melhor_objetivo_global:.2f}")

    return melhor_objetivo_global, historico_convergencia

def realizar_grid_search(repeticoes_por_teste=5):
    """Testa diferentes combinações de pc e pm para encontrar a melhor configuração."""
    valores_pc = [0.7, 0.9]
    valores_pm = [0.01, 0.05]
    arquivo = "./codigo/att48.tsp"

    print(f"Iniciando Grid Search TSP (Avaliando {len(valores_pc) * len(valores_pm)} combinações, {repeticoes_por_teste} vezes cada)...")
    print("-" * 55)
    print(f"{'p_c':<6} | {'p_m':<6} | {'Média Distância':<18} | {'Melhor Encontrado'}")
    print("-" * 55)

    melhor_configuracao = None
    melhor_media_geral = np.inf
    melhor_historico = []

    for pc in valores_pc:
        for pm in valores_pm:
            resultados = []
            historicos = []

            for _ in range(repeticoes_por_teste):
                resultado, historico = executar_ga_tsp(arquivo, pc, pm, silencioso=True)
                resultados.append(resultado)
                historicos.append(historico)

            media_resultados = np.mean(resultados)
            melhor_absoluto = np.min(resultados)

            print(f"{pc:<6} | {pm:<6} | {media_resultados:<18.2f} | {melhor_absoluto:.2f}")

            if media_resultados < melhor_media_geral:
                melhor_media_geral = media_resultados
                melhor_configuracao = (pc, pm)
                melhor_historico = historicos[np.argmin(resultados)]

    print("-" * 55)
    print(f"Melhor configuração encontrada: p_c = {melhor_configuracao[0]}, p_m = {melhor_configuracao[1]}")
    print(f"Com média de: {melhor_media_geral:.2f}")

    return melhor_configuracao, melhor_historico

def main():
    realizar_grid_search(repeticoes_por_teste=5)

if __name__ == "__main__":
    main()
