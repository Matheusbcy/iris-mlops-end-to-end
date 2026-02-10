# iris-mlops-end-to-end

Projeto de ponta a ponta para classificação com foco em **MLOps** usando o dataset Breast Cancer do scikit-learn, pipeline de treinamento em Python e aplicação web com Flask para inferência via upload de CSV.

## 📌 Visão geral

Este repositório implementa um fluxo simples de ML com as etapas:

1. **Coleta de dados** (`src/data_loading/load_data.py`)
2. **Pré-processamento** (`src/data_preprocessing/preprocess_data.py`)
3. **Engenharia de atributos** (`src/feature_engineering/engineer_features.py`)
4. **Treinamento do modelo** (`src/model_training/train_model.py`)
5. **Avaliação** (`src/model_evaluation/evaluate_model.py`)
6. **Serving** com Flask (`app/main.py`)

A aplicação recebe um arquivo CSV com as features esperadas e retorna as predições do modelo treinado.

---

## 🧱 Estrutura do projeto

```bash
.
├── app/
│   ├── main.py                    # API/UI Flask para inferência
│   └── templates/
│       └── index.html             # Interface de upload do CSV
├── src/
│   ├── data_loading/
│   │   └── load_data.py           # Carrega dataset e salva em data/raw
│   ├── data_preprocessing/
│   │   └── preprocess_data.py     # Split + imputação
│   ├── feature_engineering/
│   │   └── engineer_features.py   # Escalonamento de features
│   ├── model_training/
│   │   └── train_model.py         # Treinamento Keras + artefatos
│   └── model_evaluation/
│       └── evaluate_model.py      # Métricas no conjunto de teste
├── data/
│   ├── raw/
│   ├── preprocessed/
│   └── processed/
├── artifacts/                     # Imputer, scaler e encoder
├── models/                        # Modelo treinado (.keras)
├── metrics/                       # Métricas de treino/avaliação
├── params.yaml                    # Hiperparâmetros e configs do pipeline
├── pyproject.toml                 # Dependências do projeto
└── Dockerfile                     # Container para serving com Gunicorn
```

---

## ✅ Pré-requisitos

- Python **3.12+**
- `pip`

> Observação: o treinamento usa TensorFlow/Keras, então é necessário ter essa dependência instalada no ambiente (ela não está listada hoje no `pyproject.toml`).

---

## ⚙️ Configuração do ambiente

Na raiz do projeto:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install --upgrade pip
pip install -e .
```

Se necessário, instale também:

```bash
pip install tensorflow joblib
```

---

## 🧪 Como executar o pipeline manualmente

Antes de treinar, preencha os valores de `params.yaml` (atualmente estão vazios), por exemplo:

```yaml
train:
  learning_rate: 0.001
  hidden_layer_1_neurons: 64
  hidden_layer_2_neurons: 32
  dropout_rate: 0.2
  epochs: 50
  batch_size: 32
  random_seed: 42

preprocess_data:
  test_size: 0.2
  random_seed: 42
```

Execute as etapas na sequência:

```bash
python src/data_loading/load_data.py
python src/data_preprocessing/preprocess_data.py
python src/feature_engineering/engineer_features.py
python src/model_training/train_model.py
python src/model_evaluation/evaluate_model.py
```

Saídas esperadas:

- `data/raw/raw.csv`
- `data/preprocessed/*.csv`
- `data/processed/*.csv`
- `artifacts/*.joblib`
- `models/model.keras`
- `metrics/training.json`
- `metrics/evaluation.json`

---

## 🚀 Executar a aplicação Flask

Após gerar os artefatos de treinamento:

```bash
python app/main.py
```

A aplicação ficará disponível em:

- `http://localhost:5001`

Fluxo de uso:

1. Acesse a página inicial.
2. Faça upload de um CSV com as colunas esperadas do dataset.
3. Visualize as predições na interface.

---

## 🐳 Executar com Docker

Build da imagem:

```bash
docker build -t iris-mlops-e2e .
```

Run do container:

```bash
docker run --rm -p 5001:5001 iris-mlops-e2e
```

Servidor disponível em `http://localhost:5001`.

---

## 🔍 Possíveis melhorias

- Incluir `tensorflow` e `joblib` no `pyproject.toml`.
- Automatizar o pipeline com um orquestrador (ex.: Makefile, DVC, Airflow, Prefect).
- Adicionar testes unitários e de integração.
- Melhorar versionamento e rastreabilidade de modelos/experimentos.

---

## 📄 Licença

Defina aqui a licença do projeto (ex.: MIT, Apache-2.0).
