import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
from sklearn.impute import SimpleImputer
import os

caminho_base = os.path.dirname(os.path.abspath(__file__))
caminho_csv = os.path.join(caminho_base, '..', 'dados', 'df_consolidado.csv')

df = pd.read_csv(caminho_csv,low_memory=False)

# Variável alvo
df['voto_favoravel'] = df['tipoVoto'].apply(lambda x: 1 if x == 1.0 else 0)
y = df['aprovacao']

# Seleção de features
features = ['siglaUf', 'id_partido', 'cod_tipo', 'numero_proposicao', 'ano', 'tema']
X = df[features]
X = X.copy()
X['tema_principal'] = X['tema'].apply(lambda x: eval(x)[0] if pd.notnull(x) else 'Outros')
X_final = X[['siglaUf', 'id_partido', 'cod_tipo', 'ano', 'tema_principal']]
dados = pd.concat([X_final, y], axis=1).dropna()
X_final = dados[X_final.columns]
y = dados['aprovacao']

# Divisão das colunas
categorical_features = ['siglaUf', 'tema_principal']
numeric_features = ['id_partido', 'cod_tipo', 'ano']

# Pré-processamento com escalonamento
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42)

# Função para treinar e avaliar um modelo
def avaliar_modelo(modelo, nome_modelo):
    print(f"\n🔍 Avaliando modelo: {nome_modelo}")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', modelo)
    ])
    
    inicio = time.time()
    pipeline.fit(X_train, y_train)
    tempo_treino = time.time() - inicio

    y_pred = pipeline.predict(X_test)

    print(f"⏱ Tempo de treino e predição: {tempo_treino:.2f} segundos")
    print(f"✅ Acurácia: {accuracy_score(y_test, y_pred):.2f}")
    print("📊 Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    return pipeline

# Avaliar KNN
knn_model = avaliar_modelo(KNeighborsClassifier(n_neighbors=5), "K-Nearest Neighbors")

# Avaliar SVM
svm_model = avaliar_modelo(SVC(kernel='rbf', C=1.0), "Support Vector Machine")

