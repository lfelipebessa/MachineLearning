# 📄 Documentação dos Dados Coletados

Esta documentação descreve os arquivos `.csv` gerados a partir dos dados abertos da Câmara dos Deputados, bem como as funções responsáveis por sua criação. Os dados foram obtidos exclusivamente para proposições legislativas e suas respectivas votações.

---

## 1. Deputados
**Função:** `get_deputados()`
- **Arquivo gerado:** `deputados.csv`
- **Colunas:** `id`, `nome`, `siglaPartido`, `siglaUf`, `idLegislatura`, `urlFoto`, `email`
- **Relacionamentos:**
  - `id` → `id_deputado` nas despesas e votos

---

## 2. Partidos
**Função:** `get_partidos()`
- **Arquivo gerado:** `partidos.csv`
- **Colunas:** `id`, `sigla`, `nome`, `uri`

---

## 3. Despesas por Deputado
**Função:** `get_despesas_deputados(ano=2023)`
- **Arquivo gerado:** `despesas_2023.csv`
- **Colunas:** `id_deputado`, `nome_deputado`, `tipo_despesa`, `fornecedor`, `valor`, `data`, `documento`
- **Relacionamentos:**
  - `id_deputado` → `id` em `deputados.csv`

---

## 4. Proposições
**Função:** `get_proposicoes_por_ano(ano)`
- **Arquivo gerado:** `proposicoes_2023.csv`
- **Colunas:** `id`, `uri`, `siglaTipo`, `codTipo`, `numero`, `ano`, `ementa`
- **Relacionamentos:**
  - `id` → `id_proposicao` em votos, autores, temas

---

## 5. Votações por Proposição
**Função:** `get_votacoes_por_proposicoes_parallel(ano)`
- **Arquivo gerado:** `votacoes_proposicoes_2023.csv`
- **Colunas:** `id`, `uri`, `data`, `descricao`, `aprovacao`, `id_proposicao`
- **Relacionamentos:**
  - `id_proposicao` → `id` em `proposicoes.csv`
  - `id` → `id_votacao` em votos e orientações

---

## 6. Votos Individuais por Proposição
**Função:** `get_votos_individuais_proposicoes(ano, max_workers=10)`
- **Arquivo gerado:** `votos_individuais_proposicoes_2023.csv`
- **Colunas:** `id_votacao`, `id_deputado`, `tipoVoto`, `nome`, `siglaPartido`, `siglaUf`
- **Relacionamentos:**
  - `id_votacao` → `id` em `votacoes_proposicoes.csv`
  - `id_deputado` → `id` em `deputados.csv`

---

## 7. Orientações Partidárias por Votação
**Função:** `fetch_orientacoes(votacao_id)` (usada em paralelo)
- **Arquivo gerado:** `orientacoes_proposicoes_2023.csv`
- **Colunas:** `id_votacao`, `siglaPartido`, `orientacao`
- **Relacionamentos:**
  - `id_votacao` → `id` em `votacoes_proposicoes.csv`
  - `siglaPartido` → `sigla` em `partidos.csv`

---

## 8. Autores das Proposições
**Função:** `fetch_autores(prop_id)` (usada em paralelo)
- **Arquivo gerado:** `autores_proposicoes_2023.csv`
- **Colunas:** `id_proposicao`, `idAutor`, `nome`, `tipo`, `partido`, `uf`
- **Relacionamentos:**
  - `id_proposicao` → `id` em `proposicoes.csv`

---

## 9. Temas das Proposições
**Função:** `fetch_temas(prop_id)` (usada em paralelo)
- **Arquivo gerado:** `temas_proposicoes_2023.csv`
- **Colunas:** `id_proposicao`, `tema`
- **Relacionamentos:**
  - `id_proposicao` → `id` em `proposicoes.csv`

---

## 🔗 Recomendações de uso
- Para **análises de alinhamento ideológico**, relacione `votos_individuais_proposicoes_2023.csv` com `orientacoes_proposicoes_2023.csv` e `partidos.csv`.
- Para **mapear temáticas mais votadas**, relacione `temas_proposicoes_2023.csv` com `votacoes_proposicoes_2023.csv`.
- Para **avaliar autores mais ativos**, relacione `autores_proposicoes_2023.csv` com `votos_individuais_proposicoes_2023.csv` e `proposicoes_2023.csv`.
- Para **investigar gastos**, utilize `despesas_2023.csv` junto a `deputados.csv`.


## 🔄 Mapeamento de relacionamentos

| Tabela 1                     | Chave             | Relacionada com                  | Chave Relacionada   |
|-----------------------------|-------------------|----------------------------------|----------------------|
| `deputados.csv`            | `id`              | `votos_individuais_proposicoes` | `id_deputado`        |
| `deputados.csv`            | `id`              | `despesas_2023.csv`              | `id_deputado`        |
| `partidos.csv`             | `sigla`           | `orientacoes_proposicoes`        | `siglaPartido`       |
| `proposicoes_2023.csv`     | `id`              | `votacoes_proposicoes`           | `id_proposicao`      |
| `proposicoes_2023.csv`     | `id`              | `temas_proposicoes`              | `id_proposicao`      |
| `proposicoes_2023.csv`     | `id`              | `autores_proposicoes`            | `id_proposicao`      |
| `votacoes_proposicoes`     | `id`              | `votos_individuais_proposicoes`  | `id_votacao`         |
| `votacoes_proposicoes`     | `id`              | `orientacoes_proposicoes`        | `id_votacao`         |

