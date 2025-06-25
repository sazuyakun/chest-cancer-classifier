
# Chest Cancer Classifier

A machine learning project for chest cancer classification using MLOps practices.

## Final App Preview
https://github.com/user-attachments/assets/685f0596-f05e-481d-9ce1-59bcf9a31ef7

## Features

- Deep learning model for chest cancer classification
- MLOps pipeline with DVC and MLflow
- Flask web application for inference
- Automated CI/CD with GitHub Actions

## Will add...
- Functionality for the logging the pytorch model in mlflow tracking

## Training Methodology

> Transfer Learning on VGG-16

## Installation

```bash
pip install -r requirements.txt
```

## DVC Usage
1. Initialise dvc
```bash
dvc init
```
2. Regenerate data pipeline results by restoring the dependency graph defined among the stages in dvc.yaml
```bash
dvc repro
```

## Usage

Coming soon...

## Project Structure

```
chest-cancer-classifier/
├── src/
│   └── chest-cancer-classifier/
├── config/
├── research/
├── templates/
└── requirements.txt
```

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml

## Pipeline

1. Data Ingestion
2. Prepare Base Model
3. Training
