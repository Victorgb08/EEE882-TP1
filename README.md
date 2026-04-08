# Trabalho Prático 1 - Computação Evolucionária (EEE882)

Repositório destinado ao desenvolvimento do Trabalho Prático 1 da disciplina EEE882 - Computação Evolucionária (UFMG). O trabalho consiste na aplicação de Algoritmos Genéticos (AGs) para a resolução de dois problemas de otimização distintos.

## Estrutura do Repositório
* `/codigo`: Scripts Python contendo a implementação dos algoritmos e a geração de gráficos.
  * `n_rainhas.py`: Resolução do problema combinatório das 8-Rainhas (Representação por permutação, Cruzamento OX, Mutação Swap).
  * `rastrigin.py`: Minimização da Função Rastrigin em 10 dimensões (Codificação binária, Seleção híbrida, Grid Search de hiperparâmetros).
  * `plot_n_rainhas.py`: Gera o histograma de convergência do problema das N-Rainhas.
  * `plot_rastrigin.py`: Gera a curva de convergência do problema da Função Rastrigin.
* `/relatorio`: Imagens geradas e arquivos LaTeX para a compilação do relatório final em formato IEEE.

## Configuração do Ambiente
Este projeto foi desenvolvido utilizando a ferramenta [`uv`](https://docs.astral.sh/uv/). Basta clonar o repositório e installar a ferramenta.

## Como Executar
Todos os comandos abaixo devem ser executados na raiz. Sempre que os scripts sejam modificados, execute `uv build` para atualizar ou use a  `--refresh`:

**Problema 1: N-Rainhas**
* Para executar a bateria de testes e visualizar a média de gerações: `uv run n_rainhas`
* Para gerar o gráfico de histograma na pasta `/relatorio`: `uv run plot_n_rainhas`

**Problema 2: Função Rastrigin**
* Para executar o Grid Search e encontrar os melhores parâmetros de cruzamento e mutação: `uv run rastrigin`
* Para gerar o gráfico de curva de convergência na pasta `/relatorio`: `uv run plot_rastrigin`
