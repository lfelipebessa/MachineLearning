import pandas as pd
import numpy as np
# import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# Carregar os dados (mantendo seu código original)
df = pd.read_csv('../dados/df_consolidado.csv', dtype={
    'id_votacao': 'str',
    'id_deputado': 'int64',
    'tipoVoto': 'str',
    'siglaUf': 'str',
    'id_partido': 'float64',
    'id_proposicao': 'int64',
    'data': 'str',
    'sigla_orgao': 'str',
    'aprovacao': 'float64',
    'cod_tipo': 'int64',
    'numero_proposicao': 'int64',
    'ano': 'int64',
    'orientacao': 'str',
    'id_autor': 'float64',
    'tema': 'str'
})

# Remover linhas onde qualquer uma das colunas especificadas é NaN
colunas_para_remover_nan = [
    'id_votacao', 'id_deputado', 'tipoVoto', 'siglaUf', 'id_partido',
    'id_proposicao', 'data', 'sigla_orgao', 'aprovacao', 'cod_tipo',
    'numero_proposicao', 'ano', 'orientacao', 'id_autor', 'tema'
]
df = df.dropna(subset=colunas_para_remover_nan)

# Pré-processamento
# Converter voto em variável binária (1 = a favor, 0 = contra/abstenção)
df['voto_favoravel'] = df['tipoVoto'].apply(lambda x: 1 if x == '1.0' else 0)

# Criar variável alvo: aprovação da proposição
y = df['aprovacao']

# Selecionar features relevantes
features = ['siglaUf', 'id_partido', 'cod_tipo', 'numero_proposicao', 'ano', 'tema']
X = df[features].copy()

# Pré-processamento das features categóricas
# Para temas, vamos extrair os principais temas (primeiro da lista)
X.loc[:, 'tema_principal'] = X['tema'].apply(lambda x: eval(x)[0] if pd.notnull(x) else 'Outros')

# Selecionar colunas finais para o modelo
final_features = ['siglaUf', 'id_partido', 'cod_tipo', 'ano', 'tema_principal']
X_final = X[final_features]

# Codificar variáveis categóricas
categorical_features = ['siglaUf', 'tema_principal']
numeric_features = ['id_partido', 'cod_tipo', 'ano']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42)

# Calcular pesos das classes para lidar com desbalanceamento
classes = np.unique(y_train)
weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# Criar pipeline com pré-processamento e modelo Random Forest
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,  # Número de árvores na floresta
        max_depth=15,      # Profundidade máxima das árvores
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight=class_weights,
        n_jobs=-1         # Usar todos os cores disponíveis
    ))
])

# Opcional: Otimização de hiperparâmetros com GridSearchCV
# param_grid = {
#     'classifier__n_estimators': [50, 100, 200],
#     'classifier__max_depth': [10, 15, 20, None],
#     'classifier__min_samples_split': [2, 5, 10],
#     'classifier__min_samples_leaf': [1, 2, 4]
# }
# grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train, y_train)
# best_pipeline = grid_search.best_estimator_

# Treinar modelo
pipeline.fit(X_train, y_train)

# Avaliar modelo
y_pred = pipeline.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Feature Importance (após o pré-processamento)
# Precisamos obter os nomes das features após o OneHotEncoding
encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
encoded_cat_features = encoder.get_feature_names_out(categorical_features)
all_features = numeric_features + list(encoded_cat_features)

# Obter importância das features
importances = pipeline.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Plotar a importância das features
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importância')
plt.ylabel('Features')
plt.title('Importância das Features')
plt.gca().invert_yaxis()  # Inverter o eixo para exibir a feature mais importante no topo
plt.tight_layout()
plt.show()

# Salvar png, caso necessário
# plt.savefig('feature_importance_RandomForest.png', dpi=300)

# Salvar modelo para uso futuro
# jb.dump(pipeline, 'modelo_random_forest_aprovacao.pkl')