import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # para salvar o modelo otimizado

# Carregar os dados
df = pd.read_csv("https://media.githubusercontent.com/media/DaviRogs/CdD-ML/refs/heads/main/dados/df_consolidado.csv")

# Remover linhas com NaN nas colunas importantes
colunas_para_remover_nan = [
    'id_votacao', 'id_deputado', 'tipoVoto', 'siglaUf', 'id_partido',
    'id_proposicao', 'data', 'sigla_orgao', 'aprovacao', 'cod_tipo',
    'numero_proposicao', 'ano', 'orientacao', 'id_autor', 'tema'
]
df = df.dropna(subset=colunas_para_remover_nan)

# Criar variável alvo e features
df['voto_favoravel'] = df['tipoVoto'].apply(lambda x: 1 if x == '1.0' else 0)
y = df['aprovacao']

features = ['siglaUf', 'id_partido', 'cod_tipo', 'numero_proposicao', 'ano', 'tema']
X = df[features].copy()
X.loc[:, 'tema_principal'] = X['tema'].apply(lambda x: eval(x)[0] if pd.notnull(x) else 'Outros')

final_features = ['siglaUf', 'id_partido', 'cod_tipo', 'ano', 'tema_principal']
X_final = X[final_features]

categorical_features = ['siglaUf', 'tema_principal']
numeric_features = ['id_partido', 'cod_tipo', 'ano']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42)

# Calcular pesos das classes
classes = np.unique(y_train)
weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# Pipeline com RandomForest
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight=class_weights, n_jobs=-1))
])

# Parâmetros para RandomizedSearch
param_dist = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [None, 10, 20, 30, 40, 50],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=30,  # ajustável para tempo de execução
    scoring='f1',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Treinar busca dos melhores hiperparâmetros
random_search.fit(X_train, y_train)

# Resultados dos melhores hiperparâmetros
print("Melhores parâmetros encontrados:")
print(random_search.best_params_)
print(f"Melhor F1 score na validação cruzada: {random_search.best_score_:.4f}")

# Avaliar modelo otimizado no conjunto de teste
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nRelatório de classificação no teste:")
print(classification_report(y_test, y_pred))
print(f"Acurácia no teste: {accuracy_score(y_test, y_pred):.4f}")

# Importância das features
encoder = best_model.named_steps['preprocessor'].named_transformers_['cat']
encoded_cat_features = encoder.get_feature_names_out(categorical_features)
all_features = numeric_features + list(encoded_cat_features)

importances = best_model.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='Blues_d')
plt.title('Importância das Features (Random Forest Otimizado)')
plt.tight_layout()
plt.show()

# Salvar modelo otimizado para uso futuro
joblib.dump(best_model, 'random_forest_otimizado.joblib')
print("Modelo salvo em 'random_forest_otimizado.joblib'")
