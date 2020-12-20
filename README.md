# Trabalho final de aprendizado não supervisionado

## Objetivo

Aplicar a Análise das Componentes Principais (PCA) na base de dados ORL disponibilizada em aula (​https://github.com/lobokoch/unsupervised-learning/blob/main/dataset/ORL.rar​) e efetuar reconhecimento facial. Como método de amostragem, deve ser utilizado holdout de 70% das imagens para treinamento e 30% para os testes (as imagens tanto de treino, como de testes, devem ser sorteadas aleatoriamente).

## Problema e motivação

A princípio fiz esses código em Golang. Porém,  tive que compilar manualmente todo o OpenCV e também a versão estendida (versão 4.5). Funcionou até o momento do Resize da imagem. Causou um dump de memória que não consegui resolver. Como não queria fazer nas linguagens cuja integração estivesse bem conhecida, imaginei Rust como próxima alternativa, mas o tempo perdido pra conseguir fazer funcionar com Golang, tornou a tarefa inviável. Dessa forma, optei por uma linguagem que já estou mais familiarizado, porém, não é uma das linguagens sugeridas em sala de aula, com o objetivo de testar uma alternativa. Dessa forma optei por .net-core. E a princípio correu tudo bem. Um próximo passo poderia ser comparar performance.

## Identificação

UNIVERSIDADE REGIONAL DE BLUMENAU - INSTITUTO FURB
CURSO DE ESPECIALIZAÇÃO EM DATA SCIENCE
DISCIPLINA: ​Aprendizado de Máquina II - Aprendizado Não Supervisionado
PROFESSOR: ​Márcio Koch
