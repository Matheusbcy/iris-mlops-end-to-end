# Iris MLOps End-to-End

Projeto de exemplo de **pipeline de Machine Learning com práticas de MLOps** usando o dataset Iris, com etapas de:

- Carregamento de dados,
- Pré-processamento,
- Engenharia de atributos,
- Treinamento,
- Avaliação,
- Aplicação web para inferência.

## 📁 Estrutura do projeto

```bash
.
├── app/                         # Aplicação Flask para upload de CSV e predição
├── artifacts/                   # Artefatos de transformação/encoder
├── data/
│   ├── iris.data                # Fonte de dados
│   ├── raw/                     # Dados brutos salvos pela etapa de loading
│   ├── preprocessed/            # Saída do preprocessamento
│   └── processed/               # Dados finais para treino/avaliação
├── metrics/                     # Métricas de treinamento e avaliação
├── models/                      # Modelo treinado
├── src/
│   ├── data_loading/
│   ├── data_preprocessing/
│   ├── feature_engineering/
│   ├── model_training/
│   └── model_evaluation/
├── params.yaml                  # Hiperparâmetros e configuração
├── pyproject.toml               # Dependências e pacote
└── Dockerfile                   # Container para deploy da aplicação
```

## ✅ Requisitos

- Python **3.12+**
- `pip`

## 🚀 Instalação

```bash
pip install -e .
```

Ou usando o arquivo de build do projeto:

```bash
pip install .
```

## ⚙️ Execução do pipeline (passo a passo)

Execute os módulos na ordem abaixo:

```bash
python src/data_loading/load_data.py
python src/data_preprocessing/preprocess_data.py
python src/feature_engineering/engineer_features.py
python src/model_training/train_model.py
python src/model_evaluation/evaluate_model.py
```

Ao final, você terá:

- modelo em `models/model.joblib`
- artefatos em `artifacts/`
- métricas em `metrics/training.json` e `metrics/evaluation.json`

## 🧪 Configurações

O arquivo `params.yaml` controla parâmetros do pipeline, por exemplo:

- `preprocess_data.test_size`
- `preprocess_data.random_seed`
- `train.kernel`
- `train.C`
- `train.tol`

## 🌐 Aplicação Web

Para subir localmente:

```bash
python app/main.py
```

A aplicação fica disponível em:

- `http://localhost:5001`

## 🐳 Executando com Docker

Build da imagem:

```bash
docker build -t iris-mlops .
```

Run do container:

```bash
docker run --rm -p 5001:5001 iris-mlops
```

## 📊 Artefatos gerados

- **Dados processados**: `data/preprocessed/` e `data/processed/`
- **Modelo**: `models/model.joblib`
- **Métricas**:
  - `metrics/training.json`
  - `metrics/evaluation.json`

## 📝 Observações

Este repositório é um template educacional para organizar uma solução de ML em etapas claras e reproduzíveis.
