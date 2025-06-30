import pandas as pd
import numpy as np
# import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Carregar os dados
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
df['voto_favoravel'] = df['tipoVoto'].apply(lambda x: 1 if x == 1.0 else 0)

# Criar variável alvo: aprovação da proposição
# (assumindo que aprovacao=1.0 significa aprovada)
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

# Calcular pesos das classes
classes = np.array([0.0, 1.0])  # Converta para array NumPy
weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# Criar pipeline com pré-processamento e modelo
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight=class_weights
    ))
])

# Treinar modelo
pipeline.fit(X_train, y_train)

# Avaliar modelo
y_pred = pipeline.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Obter as importâncias das features
importances = pipeline.named_steps['classifier'].feature_importances_
feature_names = numeric_features + list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))

# Criar o gráfico de barras
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Importância')
plt.ylabel('Features')
plt.title('Importância das Features')
plt.gca().invert_yaxis()  # Inverter o eixo para exibir a feature mais importante no topo
plt.show()

# Salvar png, caso necessário
# plt.savefig('feature_importance_DecisionTree.png', dpi=300)

# Salvar modelo para uso futuro
# jb.dump(pipeline, 'modelo_aprovacao_proposicao.pkl')