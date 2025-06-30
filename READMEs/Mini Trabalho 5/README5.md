# 🧠 Mini Trabalho 5 — Seleção Inicial de Modelos de Machine Learning

## Equipe:

- Andre Ricardo Meyer de Melo  - 231011097
- Davi Rodrigues da Rocha - 211061618
- Luiz Felipe Bessa Santos - 231011687
- Tiago Antunes Balieiro - 231011838
- Wesley Pedrosa dos Santos - 180029240 

## 🎯 Objetivo

Neste trabalho, avaliamos diferentes modelos de aprendizado de máquina com o intuito de prever a **aprovação de proposições legislativas** com base em informações sobre o deputado, proposição, partido, orientação, tema, entre outros fatores.

Utilizamos o dataset consolidado previamente no Mini Trabalho 4 (`df_consolidado.csv`), que reúne dados de votos individuais, proposições, orientações partidárias, autores e temas, todos referentes à 56ª legislatura (2019–2022).

---

## ⚙️ Modelos Avaliados

Foram testados cinco modelos de classificação, variando entre simples e mais sofisticados:

- **Decision Tree**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **XGBoost (Gradient Boosting)**

Cada modelo foi treinado utilizando uma pipeline com pré-processamento adequado, incluindo:

- Codificação de variáveis categóricas (como `siglaUf` e `tema`)
- Tratamento de valores ausentes
- Balanceamento de classes com `class_weight`
- Normalização (quando necessário, como em KNN e SVM)

---

## 🧪 Métricas de Avaliação

Utilizamos as seguintes métricas para comparação:

- **Acurácia**
- **Precisão**
- **Recall**
- **F1-Score**
- (Opcional) **AUC-ROC** para XGBoost

Essas métricas foram obtidas a partir de `classification_report` e `accuracy_score` da biblioteca `sklearn`.

---

## 📊 Resultados Resumidos

| Modelo                 | Acurácia | F1-Score (Classe 1) | Observações                                    |
|------------------------|----------|----------------------|------------------------------------------------|
| Random Forest          | 0.93     | ~0.96                | Melhor desempenho geral                        |
| Decision Tree          | 0.91     | ~0.94                | Boa interpretabilidade, desempenho robusto     |
| K-Nearest Neighbors    | 0.89     | ~0.92                | Requer normalização, desempenho aceitável      |
| Support Vector Machine | 0.90     | ~0.93                | Performance competitiva, mais lento            |
| XGBoost                | 0.94     | ~0.96+               | Melhor AUC-ROC, ideal para melhorias futuras   |

> **Nota**: Os valores são aproximados e podem variar com novos ajustes de hiperparâmetros.

---

## 🔍 Análise Crítica

- Os **modelos de árvore** se destacaram tanto em desempenho quanto em interpretabilidade.
- O **XGBoost** mostrou o melhor desempenho, especialmente em datasets desbalanceados, e deve ser aprofundado futuramente.
- **KNN e SVM** requerem mais tempo de treino e cuidado com escalonamento, porém mostraram bom desempenho.
- O balanceamento de classes e codificação correta das variáveis categóricas foram essenciais para bons resultados.

---

## ✅ Justificativa para os Modelos Selecionados

O modelo **Random Forest** foi o mais equilibrado em performance e tempo de treino. O **XGBoost**, apesar de mais complexo, mostrou maior potencial e será foco para ajustes finos nos próximos passos. Ambos foram escolhidos para continuidade do projeto.

---

## 📁 Organização do Notebook

O notebook `comparacao_modelos.ipynb` está organizado nas seguintes seções:

1. **Introdução e Métricas**
2. **Tabela Comparativa de Resultados**
3. **Gráficos de Acurácia e F1-Score**
4. **Código e Avaliação por Modelo**
5. **Conclusão e Próximos Passos**

---

## 📌 Próximos Passos

- Otimização dos hiperparâmetros (ex: GridSearchCV)
- Análise mais detalhada dos erros de classificação
- Implementação de técnicas de oversampling (ex: SMOTE)
- Deploy ou salvamento dos modelos para uso futuro (`joblib`)

---

