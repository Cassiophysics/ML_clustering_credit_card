# Clusterização de Clientes de Cartões de Crédito

![capa_ml_c](https://github.com/Cassiophysics/ML_clustering_credit_card/assets/108491443/c9ce7ac3-d02d-4913-a148-5b5b3f485004)

## Teste você mesmo o modelo: [🎯 Classificador de Clientes](https://cassiophysics-ml-clustering-credit-card-streamlitapp-i9mqcy.streamlit.app/)

Este projeto tem a finalidade de agrupar titulares de cartões de crédito com características semelhantes entre si e  distintas entre outros grupos com base em seus hábitos de compra, limites de crédito, saldos e outros fatores financeiros. Essa segmentação auxilia empresas na tomada de decisões estratégicas para marketing de maneira fundamentada.

Para esta tarefa, foi usado um conjunto de dados histórico com as informações de uso do cartão de crédito de diversos clientes ativos durante o período de 6 meses. Tal amostra foi encontrada no site [KAGGLE](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).

## Motivação

O mercado de cartões de crédito está cada vez mais competitivo, com diversas instituições financeiras buscando atrair e manter clientes em um ambiente repleto de opções. Nesse contexto, é fundamental que as empresas desenvolvam estratégias de marketing eficazes para se destacarem e oferecerem propostas personalizadas aos seus clientes.

Nesse sentido, a segmentação de titulares de cartões de crédito torna-se uma abordagem estratégica essencial. Por meio da aplicação de técnicas avançadas de aprendizado não supervisionado, como o modelo de machine learning desenvolvido neste projeto, é possível identificar grupos de clientes com características e comportamentos semelhantes, permitindo uma compreensão mais profunda do perfil do público-alvo.

A principal motivação desse projeto é fornecer uma solução eficiente para a segmentação de titulares de cartões de crédito, visando auxiliar as instituições financeiras na criação de campanhas de marketing mais direcionadas e relevantes. Com a aplicação desse modelo, as empresas poderão identificar segmentos de clientes com necessidades específicas, preferências de consumo e comportamentos de compra distintos.

A segmentação precisa e eficaz permitirá que as instituições financeiras adaptem suas estratégias de marketing, oferecendo produtos e serviços personalizados, campanhas promocionais direcionadas e comunicações mais relevantes. Isso resultará em uma maior eficiência de investimento em marketing, maior satisfação dos clientes, aumento da retenção e lealdade, e, consequentemente, impactos positivos nos resultados financeiros da empresa.

Por meio desse projeto, busca-se fornecer uma solução prática e de fácil implementação para que as instituições financeiras possam explorar ao máximo as informações contidas nos dados de seus titulares de cartões de crédito, permitindo uma compreensão mais completa e precisa do seu público-alvo. O resultado será uma abordagem de marketing mais inteligente, estratégica e efetiva, que promoverá o crescimento e o sucesso das empresas no mercado altamente competitivo dos cartões de crédito.


O trabalho foi estruturado em:

## 1. Análise Exploratória

Nesta etapa foram utilizadas técnicas com o objetivo de entender o conjunto de dados o melhor possível para se ter noção de quais pré-processamentos serão necessários e quais os algoritmos que melhor se  adéquam para o problema em questão.

Algumas destas técnicas foram:

- **Estatísticas descritivas:** visando entender os dados calculamos a média, mediana, desvio padrão, quartis e outras medidas de tendência central e dispersão. Além disso, foi constatado que o Dataset possui: 8950 linhas e 18 colunas, a única coluna categórica é a CUST_ID, não existem campos duplicados, MINIMUM_PAYMENTS possui 313 valores nulos e CREDIT_LIMIT 1 valor nulo.

- **Visualização de dados:** foram utilizados histogramas, box plots e matriz de correlação, onde foi possível identificar uma abundância de valores discrepantes (outliers), os dados não estavam na mesma escala, a distribuição é assimétrica e uma forte correlação entre colunas que representam uma característica específica e as colunas que representam a frequência e atributos semelhantes desta coluna.


## 2. Pré-processamento

O objetivo desta fase é preparar os dados de forma que eles possam ser utilizados de maneira eficiente e eficaz em modelos e análises posteriores.
Para tal fim, foi realizado:
- a exclusão da coluna CUST_ID, pois não era relevante para o modelo
- exclusão do único valor nulo na coluna CREDIT_LIMIT (devido ao tamanho do Dataset não faz diferença)
- substituição dos 313 valores nulos na coluna MINIMUM_PAYMENTS pela mediana de modo a manter a maior quantidade de informações
- normalização dos dados para colocar todas as colunas na mesma escala (0 a 1)
- aplicação da transformação logarítmica para termos uma distribuição mais simétrica e suavizar a influência dos outliers
- técnicas para se descobrir a quantidade ideal de PCAs

## 3. Modelagem

Este passo consiste no processo de construção e avaliação dos modelos, para nosso caso foram utilizados três algoritmos diferentes:

- **K-Means:** funciona dividindo os dados em k grupos com base nas distâncias entre os pontos de dados e os centroides, e de forma interativa atribui cada ponto ao grupo mais próximo e recalcula os centroides até que os grupos não mudem mais. O número de k no algoritmo k-means deve ser fornecido previamente. Este número representa o número de grupos (ou clusters) em que o conjunto de dados será dividido. Escolher o número correto de k é importante para o desempenho do algoritmo e para a interpretação dos resultados.

- **DBSCAN:** algoritmo baseado em densidade que busca agrupar pontos próximos entre si e separar aqueles que estão longe uns dos outros. Ele funciona definindo dois parâmetros: eps e min_samples. O eps é utilizado para definir a distância máxima entre dois pontos para eles serem considerados vizinhos, enquanto o min_samples é o número mínimo de pontos em um eps para que um ponto seja considerado como um "centro de cluster". DBSCAN então identifica áreas de alta densidade e as marca como clusters, enquanto os pontos isolados são marcados como ruído. Para se encontrar os valores ideais de eps e min_samples foi utilizado o algoritmo NearestNeighbors, GridSearchCV e um loop aninhado.

- **Agrupamento Hierárquico Aglomerativo:** baseado em hierarquia que começa com cada ponto de dados como um cluster individual e, em seguida, combina gradualmente os clusters mais próximos até que todos os pontos de dados estejam agrupados em um único cluster. Isso é feito calculando a distância entre os clusters e escolhendo aquele com a menor distância para combinar. Esse processo é repetido até que todos os pontos de dados estejam agrupados em um único cluster ou até que o número desejado de clusters seja alcançado.

Além da plotagem de gráficos para visualizar o comportamento dos clusters em cada um dos algoritmos, as métricas utilizadas para avaliação e escolha final foram: 

- **Método "cotovelo":** consiste em plotar o valor da inércia (distortion score) que é a soma das distâncias quadráticas das amostras para o centro do cluster mais próximo, em relação ao número de clusters e escolher o ponto de torção ou "cotovelo" na curva como o melhor valor de k. Esse ponto é geralmente o ponto onde a inércia começa a diminuir mais lentamente, indicando que adicionar mais clusters não vai melhorar significativamente a qualidade do agrupamento. Quando o número de clusters aumenta, cada cluster fica menor e mais específico, permitindo que os pontos dentro de um cluster sejam mais similares entre si e menos distantes do seu centro. Isso faz com que a inércia diminua, pois, as distâncias entre as amostras e seus respectivos centros de cluster são menores, por esse motivo o valor de inércia tende a diminuir quando o número de clusters aumenta. Em outras palavras, a inércia mede a dissimilaridade dos dados dentro do cluster. Quanto menor for a inércia, menor será a dissimilaridade, portanto, melhor será o modelo de agrupamento no algoritmo k-means. No entanto, é importante evidenciar que a inércia é uma medida de dissimilaridade interna e não considera a dissimilaridade entre os clusters.

- **Índice de Silhoutte:** mede a similaridade de cada amostra com os outros pontos dentro de seu próprio cluster em relação aos outros clusters. Um valor Silhouette positivo indica que a amostra está mais próxima dos pontos dentro do seu próprio cluster do que dos outros clusters. Valores Silhouette mais próximos de 1 indicam que as amostras estão mais bem agrupadas, enquanto valores mais próximos de -1 indicam que as amostras estão mal agrupadas e talvez deveriam estar em outro cluster. Valores próximos de 0 indicam que a amostra está "na fronteira" entre dois clusters. A métrica Silhouette é útil para avaliar a qualidade do agrupamento em relação aos outros clusters, e não apenas em relação ao seu próprio cluster, e é uma boa maneira de determinar o número correto de clusters a ser escolhido.

- **Índice de Calinski-Harabasz:** mede a relação entre a variação dentro dos agrupamentos (intra-cluster) e a variação entre os agrupamentos (inter-cluster). Ou seja, ela é baseada na razão entre a soma das distâncias entre os pontos de um grupo e a média do grupo e a soma das distâncias entre cada ponto e a média geral dos dados. Quanto maior o valor desta métrica, melhor é o agrupamento, pois indica que os clusters são mais distintos e compactos, enquanto um valor baixo indica que os clusters são sobrepostos e dispersos.

- **Índice de Davies-Bouldin:** é uma medida de similaridade entre as amostras de dados em um agrupamento. É calculada como a média das distâncias entre cada ponto de dados e o centro do cluster mais próximo. Geralmente quanto menor o valor de Davies-Bouldin, melhor é o modelo de agrupamento. Isso ocorre porque o valor de Davies-Bouldin mede a similaridade entre cada cluster e seus vizinhos mais próximos, e quanto menor for esse valor, menor será a similaridade, portanto, melhor será a separação entre os clusters.

O algoritmo K-Means com 3 clusters foi a escolha mais plausível para o problema de segmentação de clientes de cartão de crédito.

## 4. Interpretação dos Clusters

Uma vez que o algoritmo e o números de clusters foram definidos, agora é hora de interpretar os clusters para que sejam utilizados de forma significativa. Para tal propósito, foram plotados gráficos de distribuições, box plots e gráficos de barras dos clusters em cada coluna de maneira individual buscando-se encontrar padrões com características similares.

O resultado ficou como:

 - **Cliente 0**: Cliente Premium

     Possui alto saldo com atualização frequente;

     Tem maior limite de cartão de crédito;
     
     Realiza quantidade de compras acima dos outros grupos;
     
     Realiza compras tanto em uma só vez como parceladas;
     
     Faz pagamento integral.

- **Cliente 1**: Cliente de Baixo Limite

    Possui menor limite de crédito e saldo entre os grupos;
    
    Mantém saldo atualizado frequentemente;
    
    Realiza compras frequentemente parceladas;
    
    Tem maior quantidade de pagamentos mínimos;
    
    Realiza pagamento integral acima da média.

- **Cliente 2**: Cliente com Poucas Compras

    Possui saldo acima da média e segundo maior limite de crédito entre os grupos;
    
    Atualiza o saldo com menor frequência;
    
    Realiza poucas compras em geral;
    
    Utiliza dinheiro adiantado de forma significativa;
    
    Realiza pagamento integral abaixo da média.

Este agrupamento auxilia de forma embasada as empresas a identificarem categorias diferentes de clientes e a partir disso ajustar estratégias de marketing adequadas para cada caso em específico.

## 5. Classificação dos Clusters

Foram testados diversos modelos para a classificação dos clientes de cartão de crédito, e o modelo LGBMClassifier apresentou os melhores resultados. Em seguida, realizou-se uma busca pelos melhores hiperparâmetros utilizando o GridSearchCV. O deploy desse modelo foi feito no Streamlit, que oferece uma interface gráfica intuitiva para facilitar a utilização do modelo.

## Impacto nos negócios:

Um modelo de machine learning para a segmentação de titulares de cartões de crédito pode trazer os seguintes benefícios financeiros e de negócios para uma empresa:

- Melhoria na aquisição de clientes, direcionando esforços de marketing e vendas para segmentos específicos e aumentando as chances de adquirir clientes com maior potencial de geração de receita.

- Redução da inadimplência por meio da identificação dos perfis de clientes com maior probabilidade de inadimplência, permitindo a adoção de medidas preventivas e a redução de riscos e perdas financeiras.

- Personalização de ofertas, produtos e serviços de acordo com as necessidades e preferências de cada segmento, aumentando a relevância das ofertas, melhorando a satisfação do cliente e impulsionando oportunidades de upselling e cross-selling.

- Otimização das taxas de juros e limites de crédito com base no entendimento do perfil de risco e comportamento de gastos dos titulares de cartões em diferentes segmentos, maximizando os ganhos financeiros e equilibrando os riscos.

- Detecção de fraudes e atividades suspeitas por meio da identificação de padrões de gastos incomuns, contribuindo para a proteção dos clientes e da empresa contra transações fraudulentas e resultando em economia de recursos e preservação da reputação.

- Aprimoramento da gestão do relacionamento com o cliente ao compreender as características e preferências dos diferentes segmentos de titulares de cartões, possibilitando o desenvolvimento de programas de fidelidade, a oferta de benefícios exclusivos e a personalização das interações, fortalecendo a fidelidade à marca.

- Tomada de decisões embasadas em dados, utilizando o modelo de machine learning para direcionar decisões estratégicas, como definição de metas de crescimento, alocação de recursos e desenvolvimento de produtos e serviços, aumentando a eficácia e a precisão das decisões.

## Identificação de melhorias para o modelo:

**Engenharia de Recursos:** Analisar as variáveis existentes de forma mais profunda e buscar oportunidades de criar novas características relevantes, como combinação de variáveis, variáveis interativas e extração de características relevantes.

**Seleção de Recursos:** Realizar uma análise cuidadosa das variáveis disponíveis e selecionar aquelas que têm maior poder discriminatório e contribuição significativa para a segmentação, ajudando a reduzir a dimensionalidade dos dados e melhorar o desempenho do modelo.

**Experimentação de Algoritmos Avançados:** Além dos algoritmos já testados, explorar algoritmos mais avançados de clusterização, como Mean Shift ou Spectral Clustering, que podem ser mais adequados para identificar clusters complexos ou em densidades irregulares nos seus dados.

**Validação Externa:** Buscar fontes externas de validação, comparando as segmentações obtidas com informações conhecidas sobre os titulares de cartões de crédito, como características demográficas ou comportamentais, para verificar se o modelo está identificando padrões relevantes e obter insights adicionais para ajustes.

**Exploração de Técnicas de Ensemble:** Aplicar técnicas de ensemble, como o ensemble de clustering, para combinar os resultados de múltiplos modelos e obter uma segmentação mais robusta, aumentando a estabilidade e a precisão.

**Monitoramento Contínuo:** Estabelecer um processo de monitoramento contínuo do desempenho do modelo em produção, acompanhando a evolução dos clusters ao longo do tempo e avaliando se as segmentações continuam relevantes. Isso permitirá identificar possíveis mudanças nos padrões de comportamento dos titulares de cartões e manter o modelo atualizado e eficaz.


