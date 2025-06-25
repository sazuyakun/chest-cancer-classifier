import dagshub
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from pathlib import Path
from urllib.parse import urlparse
import json
from chestCancerClassifier.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _valid_loader(self):
        image_size = self.config.params_image_size[:-1]  # (H, W)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        valid_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        full_dataset = datasets.ImageFolder(root=self.config.training_data)
        val_size = int(0.3 * len(full_dataset))
        train_size = len(full_dataset) - val_size

        _, self.valid_dataset = random_split(full_dataset, [train_size, val_size])
        self.valid_dataset.dataset.transform = valid_transform

        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.config.params_batch_size, shuffle=False)

    @staticmethod
    def load_model(path: Path):
        model = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        return model

    def evaluation(self):

        self.model = self.load_model(self.config.path_of_model)
        self.model.to(self.device)
        self.model.eval()

        self._valid_loader()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.valid_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                total_loss += loss.item() * inputs.size(0)
                total_correct += torch.sum(preds == labels).item()
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        self.score = (avg_loss, accuracy)
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        with open("scores.json", "w") as f:
            json.dump(scores, f, indent=4)

    def log_into_mlflow(self):
        dagshub.init(repo_owner='sazuyakun', repo_name='chest-cancer-classifier', mlflow=True)

        # Give experiment name
        mlflow.set_experiment("Chest_Cancer_Classifier_Evaluation")
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {
                    "loss": self.score[0],
                    "accuracy": self.score[1]
                }
            )

            mlflow.pytorch.autolog()
