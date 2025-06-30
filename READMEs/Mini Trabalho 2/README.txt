Projeto: Coleta de Despesas de Deputados (Câmara dos Deputados)

Este repositório contém um script em Python para coletar dados de despesas dos deputados por meio da API de Dados Abertos da Câmara dos Deputados.

1. ESTRUTURA DO REPOSITÓRIO

├── dados/
│   ├── Ano-2023.csv.zip
│   ├── Ano-2024.csv.zip
│   ├── Ano-2025.csv.zip
│   ├── votacoesObjetos_votacoesVotos_votacoesOrientacoes_votacoes_deputados.zip
│   └── coleta_dados.py
├── documentacao_legal_etica.txt
├── LICENSE
└── README.md

Descrição dos Arquivos Principais:

- dados/: pasta onde ficam os arquivos de dados brutos ou comprimidos (CSV, JSON, ZIP).
- coleta_dados.py: script Python responsável por coletar as informações das despesas de cada deputado.
- documentacao_legal_etica.txt: contém informações sobre o uso de dados, conformidade legal e aspectos éticos relacionados ao projeto.
- LICENSE: arquivo de licença do projeto (caso necessário).
- README.md: este arquivo, com a descrição do projeto e instruções de uso.

2. DESCRIÇÃO DO SCRIPT coleta_dados.py

O arquivo coleta_dados.py faz o seguinte:

- Consulta a lista de deputados por meio do endpoint da API de Dados Abertos da Câmara:
  GET https://dadosabertos.camara.leg.br/api/v2/deputados?itens=1000

- Para cada deputado encontrado, realiza uma busca de despesas utilizando:
  GET https://dadosabertos.camara.leg.br/api/v2/deputados/{idDeputado}/despesas?ano=2022&itens=50&pagina=1

- Armazena as informações coletadas (ex.: número do documento, nome do deputado, tipo de despesa, fornecedor, valor, data etc.) em uma lista de dicionários.

- Ao final, salva todos os dados em um arquivo CSV chamado despesas_deputados_2024.csv, localizado no mesmo diretório onde o script é executado.

Observação: Apesar do nome do arquivo ser despesas_deputados_2024.csv, o script está configurado para buscar dados do ano de 2022. Você pode ajustar o ano e o nome do arquivo conforme a sua necessidade no código.

3. PRÉ-REQUISITOS

- Python 3.x instalado em seu sistema.
- Biblioteca requests para realizar requisições HTTP.
- Biblioteca pandas para manipular e salvar os dados em CSV.

Você pode instalar as dependências necessárias executando:
pip install -r requirements.txt

4. COMO EXECUTAR O SCRIPT

- Clone ou baixe este repositório em sua máquina.
- Abra um terminal na pasta onde o coleta_dados.py está localizado.
- Instale as dependências (conforme descrito na seção anterior).
- Execute o script com o comando:
  python coleta_dados.py

A cada requisição, o script exibirá no console o nome do deputado e uma mensagem de status da coleta.

Ao final, um arquivo chamado despesas_deputados_2024.csv será gerado na mesma pasta.

5. NOTAS IMPORTANTES

- O script utiliza um time.sleep(0.5) após cada requisição para evitar sobrecarregar o servidor da API (boa prática de respeito à taxa de requisições).
- Caso o número de itens ou páginas seja grande, a execução pode demorar. Ajuste os parâmetros no código conforme suas necessidades:

params_despesas = {
    "ano": 2022,   # Ano da despesa
    "itens": 50,   # Quantidade de itens por página
    "pagina": 1    # Número da página
}

Se quiser coletar dados de outros anos, basta alterar o valor do parâmetro "ano" no dicionário params_despesas.

Se o script travar ou retornar muitos erros de requisição, considere aumentar o intervalo de time.sleep() ou reduzir a quantidade de itens (itens).

6. DOCUMENTAÇÃO LEGAL E ÉTICA

Consulte o arquivo documentacao_legal_etica.txt para ver detalhes sobre:

- Termos de uso da API da Câmara dos Deputados
- Conformidade com a LGPD (Lei Geral de Proteção de Dados)
- Boas práticas e recomendações éticas na manipulação de dados públicos

7. INTEGRANTES

Andre Ricardo Meyer de Melo  - 231011097
Davi Rodrigues da Rocha - 211061618
Luiz Felipe Bessa Santos - 231011687
Tiago Antunes Balieiro - 231011838
Wesley Pedrosa dos Santos - 180029240 
