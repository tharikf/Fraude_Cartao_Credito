
# Script Kaggle Fraude em Transações de Cartão de Crédito

getwd()

library(dplyr)
library(readr)
library(caret)
library(ggplot2)
library(psych)
library(randomForest)
library(pROC)
library(ROCR)
library(class)
library(doParallel)
library(performanceEstimation)
library(psych)
library(tree)
library(rpart)
library(rpart.plot)
library(data.table)
library(MASS)
library(klaR)
library(Rtsne)
library(xgboost)
library(DiagrammeR)
library(MLmetrics)
library(e1071)

options(scipen = 999)

df = read.csv('card_transdata.csv', sep = ',')
glimpse(df)

# distance from home: distancia da transacao ate sua casa
# distance from last transaction: distancia da ultima transacao
# ratio to median purchase price: razao do valor da compra pela compra mediana (valor_compra / mediana(compras))
# repeat retailer: compra em lugar repetido? (1 sim, 0 nao)
# used chip: usou chip do cartao? (1 sim, 0 nao)
# used pin number: usou numero pin do cartao? (1 sim, 0 nao)
# online order: compra online? (1 sim, 0 nao)
# fraud: é fraude? (1 sim, 0 nao)

names(df) = c('d_home', 'd_last_trans', 'ratio_purchase_median', 'repeat_retailer',
              'chip', 'pin_number', 'online', 'fraud')


glimpse(df)

#### ANALISE EXPLORATORIA ####

# verificando valores NA nas colunas
colSums(is.na(df)) # nao ha valores NA


# começando com os dados de fraude #
mean((df$fraud) * 100) # 8.74% de dados sao fraudulentos no dataset

ggplot(df, aes(x = fraud)) +
  geom_bar(stat = 'count', fill = '#B72709') +
  xlab('Fraude') +
  ylab('Quantidade') +
  scale_x_continuous(breaks = c(0, 1)) +
  scale_y_continuous(breaks = seq(0, 1000000, 250000)) +
  theme_classic()

# correlacao da fraude com as variaveis numericas (d_home, purchase/median, d_last_trans)
corPlot(df) # 0.46 de fraude com purchase/median
# 0.19 de fraude com d_home
# 0.09 de fraude com d_last_trans

calc_perc = function(x, y){
  tabela_contingencia = table(x, y)
  calculo = round(((tabela_contingencia[2, 2] / length(x)) * 100), 2)
  resultado = paste(calculo, '%', sep = '')
  print(tabela_contingencia)
  cat('\n')
  print(resultado)}


calc_perc(df$fraud, df$repeat_retailer) # 7.69% das compras repetidas resultaram em fraude

calc_perc(df$fraud, df$chip) # 2.24% das compras realizadas com chip resultaram em fraude

calc_perc(df$fraud, df$pin_number) # 0.03% das compras realizadas com o pin number resultaram em fraude

calc_perc(df$fraud, df$online) # 8.27% das coimpras realizadas online resultaram em fraude


##

# analisando a variavel d_home #
df$d_home = round(df$d_home, 4)

hist(df$d_home)# o histograma apresenta dados distribuidos à esquerda
# a maioria das compras se encontra proximo a casa do dono do cartao
boxplot(df$d_home)

quantile(df$d_home, probs = seq(0, 1, 0.10)) # 50% das compras se encontram ate 10km da casa.
# do dono do cartao. 90% se encontra ate 60km.

quantile(df$d_home, probs = seq(0.90, 1, 0.01)) # 95% se encontra há até 99.70km da casa 
# do dono do cartao. nos percentis acima disso já é > 100km.
# 98%: 177 km e 99%: 259 km

summary(df$d_home) # media de 26.62km e mediana de 9.96km


##

# analisando a variavel d_last_trans #
df$d_last_trans = round(df$d_last_trans, 4)

hist(df$d_last_trans) # dados distribuidos à esquerda
# a maioria das compras se encontra a uma distancia parecida com a compra anterior

quantile(df$d_last_trans, probs = seq(0, 1, 0.10)) # ate 90% das compras se encontram
# no maximo há 10km de distancia da ultima compra

quantile(df$d_last_trans, probs = seq(0.90, 1, 0.01)) # os percentis mais elevados
# apresentam dados mais significativos. 97%: 29km; 98%: 40km; 99%: 65km.

summary(df$d_last_trans) # media de 5km e mediana de 0.999km

##

# analisando a variavel purchase/median #
df$ratio_purchase_median = round(df$ratio_purchase_median, 4)

hist(df$ratio_purchase_median) # valores à esquerda da distribuição
# a maioria das compras se encontram proximas a mediana das compras do dono do cartao

quantile(df$ratio_purchase_median, probs = seq(0, 1, 0.10)) # ate 90% das compras
# se encontram há (4 * mediana).

quantile(df$ratio_purchase_median, probs = seq(0.90, 1, 0.01)) # os percentis mais elevados
# apresentam dados mais significativos. 97%: 7.86; 98%: 9.50; 99%: 12.79

summary(df$ratio_purchase_median) # media de 1.82 e mediana de 0.99

##

# analisando a variavel repeat retailer
mean((df$repeat_retailer) * 100) # 88.15% das compras foram feitas em estabelecimentos repetidos


##

# analisando a variavel chip
mean((df$chip) * 100) # 35.03% das compras foram feitas com chip

##

# analisando a variavel pin number
mean((df$pin_number) * 100) # 10.06% das compras foram feitas com pin number

##

# analisando a variavel online
mean((df$online) * 100) # 65.05% das compras foram feitas online

#### Análise exploratória finalizada ####

#### PRE PROCESSAMENTO ####

# removendo outliers
df = df %>%
  mutate(d_home = ifelse(d_home <= (3 * sd(d_home)), d_home, 999999)) %>%
  mutate(d_last_trans = ifelse(d_last_trans <= (3 * sd(d_last_trans)), d_last_trans, 999999)) %>%
  mutate(ratio_purchase_median = ifelse(ratio_purchase_median <= (3 * sd(ratio_purchase_median)), ratio_purchase_median, 999999))

df = df %>%
  filter(d_home != 999999) %>%
  filter(d_last_trans != 999999) %>%
  filter(ratio_purchase_median != 999999)

# padronizacao dos dados, ja que essas tres variaveis estao em escalas diferentes
# nas colunas d_home, d_last_trans, ratio_purchase_median

# antes, colocando as colunas como tipo fator
cols_factor = c('repeat_retailer', 'chip', 'pin_number', 'online', 'fraud')
df[cols_factor] = lapply(df[cols_factor], factor)

# splitando os dados
set.seed(777)
index = createDataPartition(df$fraud, p = 0.7, list = FALSE)
treino = df[index, ]
teste = df[-index, ]

# fazendo copias para padronizar
treino_normalizado = treino
teste_normalizado = teste

# funcao para normalizar os dados
func_norm = function(x, y){
  return((x - min(y)) / (max(y) - min(y)))
}

colunas = c('d_home', 'd_last_trans', 'ratio_purchase_median')

treino_normalizado[colunas] = lapply(treino_normalizado[colunas], func_norm, y = treino[colunas])
teste_normalizado[colunas] = lapply(teste_normalizado[colunas], func_norm, y = treino[colunas])

rm(colunas)

# criando matriz de resultados
resultado_geral = c()

## MODELO V1 ##
# testando um modelo de regressao logistica utilizando as variaveis 
# sem padronizacao

# criando o modelo
modelo_v1 = glm(fraud ~ ., data = treino, family = binomial(link = 'logit'))
summary(modelo_v1)

# fazendo previsao
previsao_v1 = predict(modelo_v1, teste, type = 'response')
previsao_v1 = ifelse(previsao_v1 > 0.7, 1, 0)

# criando a matriz de confusao
matriz_confusao_v1 = confusionMatrix(as.factor(previsao_v1), teste$fraud)
matriz_confusao_v1 # 97.04% de acurácia

roc_v1 = roc(teste$fraud, factor(previsao_v1, ordered = TRUE))
roc_v1 # ROC = 0.7981
plot(roc_v1, col = 'red', lwd = 3, main = 'ROC Curve V1')

resultado_v1 = c('Regressao Logistica (sem padronizacao)', '97.04%', '79.81%')


## MODELO V2 ##
# Capturando importancia das variaveis com randomForest
?randomForest
random_v2 = randomForest(fraud ~ .,
                         data = treino_normalizado,
                         ntree = 100,
                         nodesize = 10,
                         importance = TRUE)

varImpPlot(random_v2)

# variaveis mais importantes:
# na escala mean decrease accuracy: ratio_purchase_median, online, d_last_trans,
                                  # d_home, pin_number, chip, repeat_retailer

# na escala mean decrease gini: ratio_purchase_median, online, d_home, pin_number,
                                # chip, d_last_trans, repeat_retailer

# um modelo sera testado sem a variavel repeat_retailer

# modelo v2 logit sem repeat_retailer
modelo_v2 = glm(fraud ~ d_home+ d_last_trans + ratio_purchase_median + chip +
                    pin_number + online,
                  data = treino_normalizado,
                  family = binomial(link = 'logit'))

summary(modelo_v2)

previsao_v2 = predict(modelo_v2, teste_normalizado, type = 'response')
previsao_v2 = ifelse(previsao_v2 > 0.7, 1, 0)

# criando a matriz de confusao
matriz_confusao_v2 = confusionMatrix(as.factor(previsao_v2), teste_normalizado$fraud)
matriz_confusao_v2 # 96.96% de acurácia

roc_v2 = roc(teste_normalizado$fraud, factor(previsao_v2, ordered = TRUE))
roc_v2 # ROC = 0.7911
plot(roc_v2, col = 'red', lwd = 3, main = 'ROC Curve V2')

resultado_v2 = c('Regressao Logistica (variaveis definidas Random Forest)', '96.96%', '79.11%')
resultado_geral = rbind(resultado_v1, resultado_v2)


## MODELO V3 ##
# apos normalizacao das variaveis numericas
# regressao logistica com todas as variaveis

# criando o modelo
modelo_v3 = glm(fraud ~ ., data = treino_normalizado,
                family = binomial(link = 'logit'))

summary(modelo_v3)

# fazendo previsao
previsao_v3 = predict(modelo_v3, teste_normalizado, type = 'response')
previsao_v3 = ifelse(previsao_v3 > 0.7, 1, 0)

# criando a matriz de confusao
matriz_confusao_v3 = confusionMatrix(as.factor(previsao_v3), teste_normalizado$fraud)
matriz_confusao_v3 # 97.04% de acurácia

roc_v3 = roc(teste_normalizado$fraud, factor(previsao_v3, ordered = TRUE))
roc_v3 # ROC = 0.7981
plot(roc_v3, col = 'red', lwd = 3, main = 'ROC Curve V3')

resultado_v3 = c('Regressao Logistica (com padronizacao)', '97.04%', '79.81%')
resultado_geral = rbind(resultado_geral, resultado_v3)


## MODELO V4
# svm com smote e normalizacao
# Fazendo balanceamento dos dados

df_v4 = df

df_v4 = sample_n(df_v4, 400000, replace = FALSE)
set.seed(777)
index_smote = createDataPartition(df_v4$fraud, p = 0.7, list = FALSE)
treino_v4 = df_v4[index_smote, ]
teste_v4 = df_v4[-index_smote, ]


# normalizando novamente para esse modelo
treino_v4_normalizado = treino_v4
teste_v4_normalizado = teste_v4

colunas = c('d_home', 'd_last_trans', 'ratio_purchase_median')

treino_v4_normalizado[colunas] = lapply(treino_v4_normalizado[colunas], func_norm, y = treino_v4[colunas])
teste_v4_normalizado[colunas] = lapply(teste_v4_normalizado[colunas], func_norm, y = treino_v4[colunas])


rm(colunas)


# perc.over = quantidade de observacoes que vai criar na classe minoritaria
# resultado sera = (classes existentes + perc.over * classes existentes)

# perc.under = quantidade de observacoes que vai criar na classe majoritaria
# resultado sera = ((perc.over * classes existentes) * perc.under)
# ou seja, se criou 40000 na classe minoritaria, pra ter um dataset 50% x 50% terá que 
# ser criado na majoritaria 40000 + minoritaria antiga = majoritaria NOVO
df_v4_smote = smote(fraud ~ ., data = treino_v4_normalizado, k = 5, perc.over = 10, perc.under = 1.1)

table(df_v4_smote$fraud)

modelo_v4 = svm(fraud ~ ., data = df_v4_smote, type = 'C-classification', kernel = 'radial')

previsao_v4 = predict(modelo_v4, teste_v4_normalizado, type = 'response')

matriz_confusao_v4 = confusionMatrix(previsao_v4, teste_v4_normalizado$fraud)
matriz_confusao_v4 # acuracia 99.20%

roc_v4 = roc(teste_v4_normalizado$fraud, factor(previsao_v4, ordered = TRUE))
roc_v4 # ROC = 0.9954
plot(roc_v4, col = 'red', lwd = 3, main = 'ROC Curve V4')

resultado_v4 = c('SVM (com smote + normalizacao)', '99.20%', '99.54%')
resultado_geral = rbind(resultado_geral, resultado_v4)


## MODELO V5
# reg logistica com smote
# regressao logistica
modelo_v5 = glm(formula = fraud ~ ., data = df_v4_smote, family = binomial(link = 'logit'))

previsao_v5 = predict(modelo_v5, teste_v4_normalizado, type = 'response')
previsao_v5 = ifelse(previsao_v5 > 0.7, 1, 0)

matriz_confusao_v5 = confusionMatrix(as.factor(previsao_v5), teste_v4_normalizado$fraud)
matriz_confusao_v5 # 95.70%

roc_v5 = roc(teste_v4_normalizado$fraud, factor(previsao_v5, ordered = TRUE))
roc_v5 # ROC = 0.9390
plot(roc_v5, col = 'red', lwd = 3, main = 'ROC Curve V5')

resultado_v5 = c('Regressao Logistica (com smote + normalizacao)', '95.70%', '93.90%')
resultado_geral = rbind(resultado_geral, resultado_v5)


## MODELO V6
# random forest
modelo_v6 = randomForest(fraud ~ .,
                        data = df_v4_smote,
                        ntree = 100,
                        nodesize = 10)


previsao_v6 = predict(modelo_v6, teste_v4_normalizado, type = 'response')

matriz_confusao_v6 = confusionMatrix(previsao_v6, teste_v4_normalizado$fraud)
matriz_confusao_v6 # 100% acuracia

roc_v6 = roc(teste_v4_normalizado$fraud, factor(previsao_v6, ordered = TRUE))
roc_v6 # ROC = 0.9999
plot(roc_v6, col = 'red', lwd = 3, main = 'ROC Curve V6')

resultado_v6 = c('Random Forest (com smote + normalizacao)', '100%', '99.99%')
resultado_geral = rbind(resultado_geral, resultado_v6)


## MODELO V7
# decision tree sem smote
modelo_v7 = rpart(fraud ~ ., data = treino_normalizado)
printcp(modelo_v7)
rpart.plot(modelo_v7)

previsao_v7 = predict(modelo_v7, teste_normalizado, type = 'class')

matriz_confusao_v7 = confusionMatrix(previsao_v7, teste_normalizado$fraud)
matriz_confusao_v7 # 99.89% acuracia

roc_v7 = roc(teste_normalizado$fraud, factor(previsao_v7, ordered = TRUE))
roc_v7 # ROC = 0.9953
plot(roc_v7, col = 'red', lwd = 3, main = 'ROC Curve V7')

resultado_v7 = c('Decision Trees (com normalizacao)', '99.89%', '99.53%')
resultado_geral = rbind(resultado_geral, resultado_v7)


## MODELO V8
# decision tree com smote
modelo_v8 = rpart(fraud ~ ., data = df_v4_smote)
printcp(modelo_v8)
rpart.plot(modelo_v8)

previsao_v8 = predict(modelo_v8, teste_v4_normalizado, type = 'class')

matriz_confusao_v8 = confusionMatrix(previsao_v8, teste_v4_normalizado$fraud)
matriz_confusao_v8 # 95.63% acuracia

roc_v8 = roc(teste_v4_normalizado$fraud, factor(previsao_v8, ordered = TRUE))
roc_v8 # ROC = 0.9761
plot(roc_v8, col = 'red', lwd = 3, main = 'ROC Curve V8')

resultado_v8 = c('Decision Trees (com smote + normalizacao)', '95.63%', '97.61%')
resultado_geral = rbind(resultado_geral, resultado_v8)


## MODELO V9
# naive bayes sem dados balanceados

modelo_v9 = naiveBayes(fraud ~ ., data = treino_normalizado)
summary(modelo_v9)

previsao_v9 = predict(modelo_v9, teste_normalizado)

matriz_confusao_v9 = confusionMatrix(previsao_v9, teste_normalizado$fraud)
matriz_confusao_v9 # 94.60% acuracia

roc_v9 = roc(teste_normalizado$fraud, factor(previsao_v9, ordered = TRUE))
roc_v9 # ROC = 0.8516
plot(roc_v9, col = 'red', lwd = 3, main = 'ROC Curve V9')

resultado_v9 = c('Naive Bayes (com normalizacao)', '94.60%', '85.16%')
resultado_geral = rbind(resultado_geral, resultado_v9)


# MODELO 10
# naive bayes com dados balanceados

modelo_v10 = naiveBayes(fraud ~ ., df_v4_smote)
summary(modelo_v10)

previsao_v10 = predict(modelo_v10, teste_v4_normalizado)

matriz_confusao_v10 = confusionMatrix(previsao_v10, teste_v4_normalizado$fraud)
matriz_confusao_v10 # 91.25% acuracia

roc_v10 = roc(teste_v4_normalizado$fraud, factor(previsao_v10, ordered = TRUE))
roc_v10 # ROC = 0.9497
plot(roc_v10, col = 'red', lwd = 3, main = 'ROC Curve V10')

resultado_v10 = c('Naive Bayes (com smote + normalizacao)', '91.25%', '94.97%')
resultado_geral = rbind(resultado_geral, resultado_v10)


## MODELO V11
# xgboost com normalizacao

modelo_v11 = xgboost(data = data.matrix(treino_v4_normalizado[, 1:7]),
                     label = data.matrix(treino_v4_normalizado[, 8]),
                     nround = 20,
                     nthread = 3,
                     max_depth = 15,
                     objective = 'binary:logistic')


importance_matrix = xgb.importance(model = modelo_v11)
xgb.plot.importance(importance_matrix = importance_matrix)

pred = predict(modelo_v11, data.matrix(teste_v4_normalizado[, 1:7]))
previsao_v11 = as.numeric(pred > 0.7)

matriz_confusao_v11 = confusionMatrix(as.factor(previsao_v11), teste_v4_normalizado$fraud)
matriz_confusao_v11 # 99.99% acuracia

roc_v11 = roc(teste_v4_normalizado$fraud, factor(previsao_v13, ordered = TRUE))
roc_v11 # ROC = 0.9999
plot(roc_v11, col = 'red', lwd = 3, main = 'ROC Curve V11')

resultado_v11 = c('Xgboost (com normalizacao)', '99.99%', '99.99%')
resultado_geral = rbind(resultado_geral, resultado_v11)

#
resultado_geral_df = as.data.frame(resultado_geral, row.names = TRUE)
names(resultado_geral_df) = c('Implementacao_Algoritmo', 'Acuracia', 'ROC')
















