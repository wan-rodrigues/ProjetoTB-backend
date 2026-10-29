# 🫁 Pulm.io | API de Triagem de Tuberculose com Inteligência Artificial

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

## 📌 Sobre o projeto

O **Pulm.io** é um projeto experimental que explora o uso de Inteligência Artificial como ferramenta de apoio à triagem de tuberculose (TB).

A proposta consiste em combinar duas fontes de informação: sintomas informados pelo usuário e imagens de radiografias do tórax.

Este repositório contém o **backend desenvolvido em Python com FastAPI**, responsável por receber os dados, processar as imagens, executar os modelos de Machine Learning e retornar as pontuações geradas.

> ⚠️ **Aviso:** este é um protótipo experimental. Seus resultados não são clinicamente validados e não devem ser utilizados para diagnosticar, descartar tuberculose ou orientar decisões médicas.

## 🎯 Objetivo

Investigar a integração de diferentes abordagens de aprendizado de máquina em uma API, utilizando informações clínicas estruturadas e imagens radiográficas.

O projeto envolve conceitos de classificação supervisionada, visão computacional, processamento de dados e desenvolvimento de serviços web.

## ✨ Funcionalidades

- Recebimento de informações sobre sintomas por meio de formulário.
- Upload de imagens de radiografias do tórax.
- Pré-processamento das imagens para entrada no modelo.
- Classificação baseada em sintomas com Random Forest.
- Análise de imagens com rede neural implementada em TensorFlow/Keras.
- Combinação ponderada das saídas dos dois modelos.
- Disponibilização dos resultados por meio de uma API REST.
- Documentação interativa dos endpoints com Swagger UI, fornecida pelo FastAPI.

## 🛠️ Tecnologias utilizadas

| Tecnologia | Aplicação |
|---|---|
| Python | Desenvolvimento da aplicação |
| FastAPI | Construção da API REST |
| Scikit-learn | Modelo de classificação Random Forest |
| TensorFlow/Keras | Modelo de análise de imagens |
| NumPy | Manipulação de arrays e dados numéricos |
| Pillow | Leitura e processamento de imagens |
| Joblib | Carregamento do modelo de sintomas |
| Uvicorn | Servidor ASGI para execução da API |

## 🧠 Funcionamento da solução

O backend utiliza dois modelos independentes:

### 1. Modelo de sintomas — Random Forest

Recebe 13 informações relacionadas a sintomas e manifestações clínicas, como febre persistente, tosse, suores noturnos, cansaço e perda de apetite.

Esses dados são organizados em um vetor numérico e encaminhados ao modelo de classificação para gerar uma pontuação associada à classe positiva.

### 2. Modelo de imagens — Rede Neural

Recebe uma imagem de radiografia, que passa pelas seguintes etapas:

1. Conversão para RGB.
2. Redimensionamento para 300 × 300 pixels.
3. Normalização dos pixels para o intervalo de 0 a 1.
4. Conversão para um array NumPy.
5. Execução da inferência com o modelo Keras.

### 3. Combinação dos resultados

As pontuações obtidas são combinadas usando uma média ponderada:

```python
prob_final = (prob_sintomas * 0.4) + (prob_imagem * 0.6)
```

O modelo de sintomas possui peso de 40%, enquanto o modelo de imagem possui peso de 60%.

A saída combinada constitui um escore experimental, não uma probabilidade clínica calibrada.

## 📂 Estrutura do projeto

```text
ProjetoTB-backend/
├── models/
│   ├── MD_ETB.h5
│   └── MD_RF.joblib
├── main.py
└── .gitattributes
```

**Arquivos principais:**

- `main.py`: inicialização da API, definição das rotas, pré-processamento de imagens e integração dos modelos.
- `MD_ETB.h5`: modelo de análise de imagens.
- `MD_RF.joblib`: modelo de classificação baseado em sintomas.

## 🔌 Endpoints da API

### GET `/`

Verifica se a aplicação está respondendo.

**Exemplo de resposta:**

```json
{
  "status": "API funcionando!"
}
```

### POST `/analisar`

Recebe dados de sintomas e uma imagem radiográfica utilizando `multipart/form-data`.

**Entradas:**

- 13 campos numéricos representando informações sobre sintomas.
- `raiox`: arquivo de imagem para análise.

**Formato da resposta:**

```json
{
  "probabilidade_sintomas": 0.0,
  "probabilidade_imagem": 0.0,
  "probabilidade_final": 0.0
}
```

Os números acima são ilustrativos e representam apenas o formato da resposta.

## 🚀 Como executar localmente

### Pré-requisitos

- Python instalado.
- Git.
- Arquivos dos modelos disponíveis na pasta `models/`.
- Dependências Python compatíveis com os modelos serializados.

### 1. Clone o repositório

```bash
git clone https://github.com/wan-rodrigues/ProjetoTB-backend.git
cd ProjetoTB-backend
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv
```

No Windows:

```bash
venv\Scripts\activate
```

No Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install fastapi uvicorn python-multipart numpy joblib tensorflow pillow scikit-learn
```

As versões utilizadas devem ser compatíveis com os arquivos de modelos treinados.

### 4. Execute a API

```bash
uvicorn main:app --reload
```

A aplicação estará disponível em:

`http://127.0.0.1:8000`

### 5. Acesse a documentação interativa

Abra no navegador:

`http://127.0.0.1:8000/docs`

Na interface Swagger UI, é possível explorar os endpoints e enviar requisições de teste.

## 💡 Aprendizados

O desenvolvimento deste projeto permitiu aplicar conhecimentos relacionados a:

- Construção de APIs REST com FastAPI.
- Integração de modelos de Machine Learning em aplicações backend.
- Utilização de modelos de classificação e redes neurais.
- Pré-processamento de imagens digitais.
- Manipulação de dados com NumPy.
- Serialização e carregamento de modelos treinados.
- Integração entre diferentes tecnologias de Inteligência Artificial.


## 👩‍💻 Desenvolvimento

**Wanuelle Rodrigues Gonçalves**

Desenvolvimento e integração do backend com modelos de Machine Learning.

**Amanda Vasconcelo Sales**

Desenvolvimento do frontend do projeto

[GitHub — @wan-rodrigues](https://github.com/wan-rodrigues)

---

*Projeto experimental desenvolvido para fins acadêmicos e de aprendizagem, sem finalidade de diagnóstico clínico.*
