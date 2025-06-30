import streamlit as st
import pandas as pd
import pickle

@st.cache(allow_output_mutation=True)
def load_pipeline():
    # aqui salvamos pipeline completo (preprocessor + model) em 'pipeline.pkl'
    with open("pipeline.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()

st.title("Previsão de Aprovação de Projeto de Lei")
st.write("Preencha os dados abaixo e clique em **Prever**:")

# --- FORMULÁRIO DE INPUTS ---
uf = st.selectbox("Sigla UF", ["SP", "RJ", "MG", "CE", "PB", "…"])
partido = st.number_input("ID Partido", min_value=1, step=1)
cod_tipo = st.number_input("Código do Tipo", min_value=1, step=1)
numero = st.number_input("Número da Proposição", min_value=1, step=1)
ano = st.number_input("Ano", min_value=2000, max_value=2025, step=1)
tema_raw = st.text_input(
    "Tema (lista em formato Python)", 
    value="['Saúde', 'Educação']"
)

# Extrai tema_principal igual a notebook
def extrair_tema_principal(x):
    try:
        lst = eval(x)
        return lst[0] if isinstance(lst, (list, tuple)) and lst else "Outros"
    except:
        return "Outros"

tema_principal = extrair_tema_principal(tema_raw)

# --- BOTÃO DE PREDIÇÃO ---
if st.button("Prever"):
    # monta um DataFrame do seu único registro
    df = pd.DataFrame([{
        "siglaUf": uf,
        "id_partido": partido,
        "cod_tipo": cod_tipo,
        "numero_proposicao": numero,
        "ano": ano,
        "tema_principal": tema_principal
    }])

    # roda pipeline completo (pré-process + model)
    pred = pipeline.predict(df)[0]
    prob = pipeline.predict_proba(df)[0][pred]

    st.markdown(
        f"**Resultado:** {'✅ Aprovado' if pred==1 else '❌ Não aprovado'}  \n"
        f"**Confiança:** {prob:.2%}"
    )
