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
Este projeto foi desenvolvido utilizando a distribuição Anaconda. Para garantir a reprodutibilidade, siga os passos abaixo para configurar o ambiente:

1. Clone este repositório.
2. Crie o ambiente virtual isolado com o Python 3.13: `conda create --name ce_tp1 python=3.13`
3. Ative o ambiente: `conda activate ce_tp1`
4. Instale as dependências numéricas e de visualização: `conda install numpy matplotlib`

## Como Executar
Todos os comandos abaixo devem ser executados a partir do diretório `/codigo`.

**Problema 1: N-Rainhas**
* Para executar a bateria de testes e visualizar a média de gerações: `python n_rainhas.py`
* Para gerar o gráfico de histograma na pasta `/relatorio`: `python plot_n_rainhas.py`

**Problema 2: Função Rastrigin**
* Para executar o Grid Search e encontrar os melhores parâmetros de cruzamento e mutação: `python rastrigin.py`
* Para gerar o gráfico de curva de convergência na pasta `/relatorio`: `python plot_rastrigin.py`
