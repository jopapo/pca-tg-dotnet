# Trabalho final de aprendizado não supervisionado

## Objetivo

Aplicar a Análise das Componentes Principais (PCA) na base de dados ORL disponibilizada em aula (​https://github.com/lobokoch/unsupervised-learning/blob/main/dataset/ORL.rar​) e efetuar reconhecimento facial. Como método de amostragem, deve ser utilizado holdout de 70% das imagens para treinamento e 30% para os testes (as imagens tanto de treino, como de testes, devem ser sorteadas aleatoriamente).

## Problema e motivação

A princípio fiz esse código em Golang. Porém,  tive que compilar manualmente todo o OpenCV e também a versão estendida (versão 4.5). Funcionou até o momento do Resize da imagem. Causou um dump de memória que não consegui resolver. Como não queria fazer nas linguagens cuja integração estivesse bem conhecida, imaginei Rust como próxima alternativa, mas o tempo perdido pra conseguir fazer funcionar com Golang, tornou a tarefa inviável. Dessa forma, optei por uma linguagem que já estou mais familiarizado, porém, não é uma das linguagens sugeridas em sala de aula, com o objetivo de testar uma alternativa. Dessa forma optei por .net-core. E a princípio correu tudo bem. Um próximo passo poderia ser comparar performance.

Curiosamente, alguns métodos disponíveis para a versão em Java não estão para C#. Dessa forma foi necessário pesquisar alternativas (Uma delas foi a `get` da classe `Mat` e o acesso ao `double` dentro da primeira posição do `array`). Dessa forma foi criada a extensão [PCAExtensions](pca/PCAExtensions.cs) com base na questão postada por [Bartosz Rachwal no StackOverflow](https://stackoverflow.com/questions/32255440/how-can-i-get-and-set-pixel-values-of-an-emgucv-mat-image) e utilizado o método `GetDoubleValue`. Outro método inexistente é o `ColRange`.


## Como rodar

Apenas é necessário ter o Visual Studio Code com o OmniSharp instalado. Na primeira execução o dotnet já vai baixar todas as dependências necessárias (inclusive a runtime do OpenCV).

Ou rode o comando:
`dotnet run`

## Mais detalhes

O programa roda com verbosidade INFO por padrão. Para mudar e apresentar mais detalhes, bem como imprimir as eigenfaces, somente é necessário diminuir o nível para Debug nesta linha:
`.Configure<LoggerFilterOptions>(options => options.MinLevel = LogLevel.Information)`

## Resumo da acurácia obtida entre 10 e 20 componentes principais
      10 componentes principais, acurácia: 68.46153846153847
      11 componentes principais, acurácia: 76.15384615384615
      12 componentes principais, acurácia: 84.61538461538461
      13 componentes principais, acurácia: 86.92307692307692
      14 componentes principais, acurácia: 93.07692307692308
      15 componentes principais, acurácia: 95.38461538461539
      16 componentes principais, acurácia: 95.38461538461539
      17 componentes principais, acurácia: 96.15384615384616
      18 componentes principais, acurácia: 96.92307692307692
      19 componentes principais, acurácia: 97.6923076923077
      20 componentes principais, acurácia: 97.6923076923077

## Identificação

UNIVERSIDADE REGIONAL DE BLUMENAU - INSTITUTO FURB
CURSO DE ESPECIALIZAÇÃO EM DATA SCIENCE
DISCIPLINA: ​Aprendizado de Máquina II - Aprendizado Não Supervisionado
PROFESSOR: ​Márcio Koch
