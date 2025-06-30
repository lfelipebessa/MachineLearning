# Objetivo

O objetivo desta etapa foi estabelecer um ponto de referência (baseline) para a tarefa de classificação. Para isso, foram utilizados dois modelos simples:

- DummyClassifier: um modelo que faz previsões aleatórias ou sempre escolhe a classe mais frequente. Serve como controle para avaliar se os modelos mais elaborados realmente aprendem padrões dos dados.

- Regressão Logística: um modelo supervisionado linear simples, que costuma funcionar bem em problemas binários.

# Preparação dos dados

O dataset final foi dividido em treino e teste utilizando a função train_test_split do scikit-learn, com 80% dos dados para treino e 20% para teste, e estratificação com base na variável alvo para manter a proporção das classes.

A variável alvo (aprovada) é binária:

- 1 = proposição aprovada

- 0 = proposição rejeitada ou arquivada

# Resultados Obtidos

| Modelo              | Acurácia | Precisão | Recall | F1-Score |
|---------------------|----------|----------|--------|----------|
| DummyClassifier     | 0.56     | 0.32     | 0.57   | 0.41     |
| Regressão Logística | 0.67     | 0.68     | 0.67   | 0.64     |


# Análise

O DummyClassifier confirmou o desbalanceamento entre as classes, já que seu desempenho ficou em torno de 56%, apenas por prever a classe mais comum (aprovação). Como esperado, ele não consegue capturar nenhum padrão útil nos dados.

Já a Regressão Logística apresentou desempenho um pouco superior, com cerca de 67% de acurácia e equilíbrio entre precisão e recall. Isso indica que mesmo um modelo linear simples já consegue capturar características relevantes para a previsão de aprovação de proposições.

Esses resultados servirão como referência para comparar modelos mais complexos nas próximas etapas do projeto.
