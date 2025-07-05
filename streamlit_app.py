import streamlit as st
import pandas as pd
import pickle

# ─── 0. Carrega e prepara dicionário de autores ───────────────────────────────
df_deputados = pd.read_csv("deputados_legislatura_56.csv")
df_unique    = df_deputados.drop_duplicates(subset="id_deputado")
autor_to_id  = dict(sorted(zip(df_unique["nome_deputado"], df_unique["id_deputado"])))

# ─── 1. Dicionários de Partido, UF e Tema ──────────────────────────────────────
sigla_to_id_partido = {
    'AVANTE':    36898,
    'CIDADANIA': 37905,
    'MDB':       36899,
    'NOVO':      37901,
    'PCdoB':     36779,
    'PDT':       36786,
    'PL':        37906,
    'PODE':      36896,
    'PP':        37903,
    'PRD':       38010,
    'PSB':       36832,
    'PSD':       36834,
    'PSDB':      36835,
    'PSOL':      36839,
    'PT':        36844,
}

id_partido_to_info = {
    36898: ('AVANTE',    'Avante'),
    37905: ('CIDADANIA', 'Cidadania'),
    36899: ('MDB',       'Movimento Democrático Brasileiro'),
    37901: ('NOVO',      'Partido Novo'),
    36779: ('PCdoB',     'Partido Comunista do Brasil'),
    36786: ('PDT',       'Partido Democrático Trabalhista'),
    37906: ('PL',        'Partido Liberal'),
    36896: ('PODE',      'Podemos'),
    37903: ('PP',        'Progressistas'),
    38010: ('PRD',       'Partido Renovação Democrática'),
    36832: ('PSB',       'Partido Socialista Brasileiro'),
    36834: ('PSD',       'Partido Social Democrático'),
    36835: ('PSDB',      'Partido da Social Democracia Brasileira'),
    36839: ('PSOL',      'Partido Socialismo e Liberdade'),
    36844: ('PT',        'Partido dos Trabalhadores'),
}

uf_to_nome = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AP": "Amapá",
    "AM": "Amazonas",
    "BA": "Bahia",
    "CE": "Ceará",
    "DF": "Distrito Federal",
    "ES": "Espírito Santo",
    "GO": "Goiás",
    "MA": "Maranhão",
    "MT": "Mato Grosso",
    "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais",
    "PA": "Pará",
    "PB": "Paraíba",
    "PR": "Paraná",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul",
    "RO": "Rondônia",
    "RR": "Roraima",
    "SC": "Santa Catarina",
    "SP": "São Paulo",
    "SE": "Sergipe",
    "TO": "Tocantins",
}

tema_to_label = {
    "Meio Ambiente e Desenvolvimento Sustentável": "Meio Ambiente e Desenvolvimento Sustentável",
    "Administração Pública":                       "Administração Pública",
    "Saúde":                                       "Saúde",
    "Educação":                                    "Educação",
    "Infraestrutura":                              "Infraestrutura",
}

# ─── 2. Carrega pipeline ─────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    with open("modelo_random_forest_20250630_1632.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()

# ─── 3. Interface ─────────────────────────────────────────────────────────────────
st.title("Previsão de Aprovação de Projeto de Lei")

# a) UF
uf = st.selectbox(
    "UF",
    list(uf_to_nome.keys()),
    format_func=lambda x: f"{x} — {uf_to_nome[x]}"
)

# b) Partido
partido_options = [
    f"{sigla} — {nome}" for _, (sigla, nome) in id_partido_to_info.items()
]
partido_escolha = st.selectbox("Partido", partido_options)
sigla_partido   = partido_escolha.split(" — ")[0]
id_partido      = sigla_to_id_partido[sigla_partido]

# c) Autor da Proposição
autor = st.selectbox("Autor da Proposição", list(autor_to_id.keys()))
id_autor = autor_to_id[autor]

# d) Tipo de proposição
cod_tipo_to_desc = {
    1: "Emenda",
    2: "Projeto de Lei",
    3: "Medida Provisória",
}
tipo_desc = st.selectbox("Tipo de proposição", list(cod_tipo_to_desc.values()))
cod_tipo  = next(k for k, v in cod_tipo_to_desc.items() if v == tipo_desc)

# e) Número e Ano
numero = st.number_input("Número da proposição", min_value=1, step=1)
ano    = st.number_input("Ano (2019-2022)", min_value=1990, max_value=2030, value=2025)

# f) Tema principal
tema = st.selectbox("Tema principal", list(tema_to_label.keys()))
tema_principal = tema_to_label[tema]

# ─── 4. Previsão ─────────────────────────────────────────────────────────────────
if st.button("Prever aprovação"):
    X_new = pd.DataFrame([{
        "siglaUf":           uf,
        "id_partido":        id_partido,
        "id_autor":          id_autor,
        "cod_tipo":          cod_tipo,
        "numero_proposicao": numero,
        "ano":               ano,
        "tema_principal":    tema_principal
    }])
    pred = pipeline.predict(X_new)[0]
    prob = pipeline.predict_proba(X_new)[0][pred]
    label = "✅ Aprovado" if pred == 1 else "❌ Não aprovado"
    st.success(f"{label} (confiança: {prob:.2%})")
