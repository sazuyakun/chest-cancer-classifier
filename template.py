import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_NAME = "chest-cancer-classifier"

PROJECT_STRUCTURE = [
    ".github/workflows/.gitkeep",

    f"src/{PROJECT_NAME}/__init__.py",
    f"src/{PROJECT_NAME}/components/__init__.py",
    f"src/{PROJECT_NAME}/utils/__init__.py",
    f"src/{PROJECT_NAME}/config/__init__.py",
    f"src/{PROJECT_NAME}/config/configuration.py",
    f"src/{PROJECT_NAME}/pipeline/__init__.py",
    f"src/{PROJECT_NAME}/entity/__init__.py",
    f"src/{PROJECT_NAME}/constants/__init__.py",

    "config/config.yaml",
    "research/trials.ipynb",
    "templates/index.html",
    
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
]

for filepath in PROJECT_STRUCTURE:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for the file: {filename}")

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, 'w') as f:
            pass
            logging.info(f"Created file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}, skipping creation.")
