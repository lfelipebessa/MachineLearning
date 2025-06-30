import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


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
df['voto_favoravel'] = df['tipoVoto'].apply(lambda x: 1 if x == '1.0' else 0)

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
    X_final, y, test_size=0.2, random_state=42, stratify=y)

# Calcular balanceamento de classes
class_weights = class_weight.compute_sample_weight('balanced', y_train)

# Opção 1: Gradient Boosting do sklearn
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1)
    ))
])

param_dist = {
    'classifier__n_estimators': randint(100, 500),
    'classifier__learning_rate': uniform(0.01, 0.3),
    'classifier__max_depth': randint(3, 10),
    'classifier__subsample': uniform(0.6, 0.4),
    'classifier__colsample_bytree': uniform(0.6, 0.4),
    'classifier__min_child_weight': randint(1, 10),
    'classifier__gamma': uniform(0, 0.5)
}

# Validação cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# RandomizedSearchCV
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    scoring='roc_auc',
    n_jobs=-1,
    cv=cv,
    verbose=1,
    random_state=42
)

# Executar busca
search.fit(X_train, y_train, classifier__sample_weight=class_weights)

# Melhor modelo
best_model = search.best_estimator_

# Avaliação final
y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]

print("\n🔍 MELHOR COMBINAÇÃO DE PARÂMETROS:")
print(search.best_params_)

print("\n📊 DESEMPENHO APÓS OTIMIZAÇÃO:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_best):.2f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_best):.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_best))

# Feature Importance
try:
    importances = best_model.named_steps['classifier'].feature_importances_
    feature_names = (
        numeric_features + 
        list(best_model.named_steps['preprocessor']
            .named_transformers_['cat']
            .get_feature_names_out(categorical_features))
    )
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importância')
    plt.ylabel('Features')
    plt.title('Importância das Features (XGBoost Otimizado)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Erro ao plotar importância das features:", str(e))

