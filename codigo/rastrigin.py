import random
import numpy as np

# Parâmetros do Problema
N_DIM = 10
BOUNDS = [-5.12, 5.12]
MAX_EVALS = 100000

# Parâmetros da Representação Binária
BITS_PER_VAR = 16  # Precisão: 16 bits por dimensão
CHROMOSOME_LENGTH = N_DIM * BITS_PER_VAR

# Hiperparâmetros fixos
POP_SIZE = 100

def decodificar(individuo):
    """Converte o cromossomo binário em um vetor de números reais."""
    x_real = []
    # Divide o cromossomo em blocos de BITS_PER_VAR
    for i in range(N_DIM):
        bloco_binario = individuo[i * BITS_PER_VAR : (i + 1) * BITS_PER_VAR]
        # Converte binário para decimal inteiro
        valor_inteiro = 0
        for bit in bloco_binario:
            valor_inteiro = (valor_inteiro << 1) | bit

        # Mapeia o inteiro para o intervalo contínuo [-5.12, 5.12]
        max_int = (2 ** BITS_PER_VAR) - 1
        valor_real = BOUNDS[0] + (valor_inteiro / max_int) * (BOUNDS[1] - BOUNDS[0])
        x_real.append(valor_real)
    return x_real

def funcao_rastrigin(x):
    """Avalia a função Rastrigin para n dimensões."""
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def aptidao_para_selecao(valor_objetivo):
    """Inverte o valor objetivo para uso na Roleta (onde maior é melhor)."""
    return 1.0 / (1.0 + valor_objetivo)

def gerar_populacao_inicial(tamanho):
    """Gera população inicial com bits aleatórios."""
    return [[random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)] for _ in range(tamanho)]

def selecao_torneio(populacao, aptidoes, k=3):
    """Seleciona o melhor indivíduo entre k sorteados aleatoriamente."""
    selecionados = random.sample(list(zip(populacao, aptidoes)), k)
    return max(selecionados,key=lambda x: x[1])[0]

def selecao_roleta(populacao, aptidoes):
    """Seleciona um indivíduo proporcionalmente à sua aptidão."""
    soma_aptidoes = sum(aptidoes)
    aptidoes_normalizadas = [a/soma_aptidoes for a in aptidoes]
    selecionado = np.random.choice(range(POP_SIZE),p=aptidoes_normalizadas)
    return populacao[selecionado]

def selecao_hibrida(populacao, aptidoes):
    """50% Roleta, 50% Torneio conforme roteiro."""
    if random.random() < 0.5:
        return selecao_roleta(populacao, aptidoes)
    else:
        return selecao_torneio(populacao, aptidoes)

def cruzamento_1ponto_por_variavel(pai1, pai2, pc):
    """Aplica 1 ponto de corte separadamente para o bloco de bits de cada variável."""
    if random.random() > pc:
        return list(pai1), list(pai2)

    filho1, filho2 = [], []
    for i in range(N_DIM):
        inicio = i * BITS_PER_VAR
        fim = (i + 1) * BITS_PER_VAR
        gene_pai1 = pai1[inicio:fim]
        gene_pai2 = pai2[inicio:fim]

        ponto_corte = random.randint(1, BITS_PER_VAR - 1)

        gene_filho1 = gene_pai1[:ponto_corte] + gene_pai2[ponto_corte:]
        gene_filho2 = gene_pai2[:ponto_corte] + gene_pai1[ponto_corte:]

        filho1.extend(gene_filho1)
        filho2.extend(gene_filho2)

    return filho1, filho2

def mutacao_bit_flip(individuo, pm):
    """Percorre cada bit e inverte com base na probabilidade de mutação (pm)."""
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < pm:
            individuo[i] = 1 - individuo[i]

def executar_ag_rastrigin(pc, pm, silencioso=True):
    """Executa o Algoritmo Genético para a Função Rastrigin."""
    populacao = gerar_populacao_inicial(POP_SIZE)
    avaliacoes = 0
    geracao = 0
    melhor_objetivo_global = np.inf
    historico_convergencia = []

    while avaliacoes < MAX_EVALS:
        # Avaliação
        fenotipos = [decodificar(ind) for ind in populacao]
        valores_objetivo = [funcao_rastrigin(x) for x in fenotipos]

        # Inverte função objetivo para a aptidão
        aptidoes = [aptidao_para_selecao(v)  for v in valores_objetivo]
        avaliacoes += POP_SIZE

        # Rastreia o melhor da geração atual
        melhor_valor_geracao = min(valores_objetivo)
        if melhor_valor_geracao < melhor_objetivo_global:
            melhor_objetivo_global = melhor_valor_geracao

        historico_convergencia.append(melhor_objetivo_global)

        nova_populacao = []

        # Elitismo: preserva o melhor indivíduo absoluto (menor valor na função Rastrigin)
        melhor_idx = np.argmin(valores_objetivo)
        nova_populacao.append(populacao[melhor_idx])

        while len(nova_populacao) < POP_SIZE:
            pai1 = selecao_hibrida(populacao, aptidoes)
            pai2 = selecao_hibrida(populacao, aptidoes)

            filho1, filho2 = cruzamento_1ponto_por_variavel(pai1, pai2, pc)

            mutacao_bit_flip(filho1, pm)
            mutacao_bit_flip(filho2, pm)

            nova_populacao.extend([filho1, filho2])

        populacao = nova_populacao[:POP_SIZE]
        geracao += 1

    if not silencioso:
        print(f"Melhor valor Rastrigin encontrado: {melhor_objetivo_global:.6f}")

    return melhor_objetivo_global, historico_convergencia

def realizar_grid_search(repeticoes_por_teste=5):
    """Testa diferentes combinações de pc e pm para encontrar a melhor configuração."""
    valores_pc = [0.7, 0.8, 0.9]
    valores_pm = [0.01, 0.05, 0.1]

    print(f"Iniciando Grid Search (Avaliando {len(valores_pc) * len(valores_pm)} combinações, {repeticoes_por_teste} vezes cada)...")
    print("-" * 55)
    print(f"{'p_c':<6} | {'p_m':<6} | {'Média Rastrigin':<18} | {'Melhor Encontrado'}")
    print("-" * 55)

    melhor_configuracao = None
    melhor_media_geral = np.inf
    melhor_historico = []

    for pc in valores_pc:
        for pm in valores_pm:
            resultados = []
            historicos = []

            for _ in range(repeticoes_por_teste):
                resultado, historico = executar_ag_rastrigin(pc, pm, silencioso=True)
                resultados.append(resultado)
                historicos.append(historico)

            media_resultados = np.mean(resultados)
            melhor_absoluto = np.min(resultados)

            print(f"{pc:<6} | {pm:<6} | {media_resultados:<18.6f} | {melhor_absoluto:.6f}")

            if media_resultados < melhor_media_geral:
                melhor_media_geral = media_resultados
                melhor_configuracao = (pc, pm)
                # Salva o histórico da melhor execução desta configuração
                melhor_historico = historicos[np.argmin(resultados)]

    print("-" * 55)
    print(f"Melhor configuração encontrada: p_c = {melhor_configuracao[0]}, p_m = {melhor_configuracao[1]}")
    print(f"Com média de: {melhor_media_geral:.6f}")

    return melhor_configuracao, melhor_historico

def main():
    realizar_grid_search(repeticoes_por_teste=5)

if __name__ == "__main__":
    main()
