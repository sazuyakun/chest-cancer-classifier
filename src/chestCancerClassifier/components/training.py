from chestCancerClassifier.config.configuration import TrainingConfig
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


    def get_base_model(self):
        self.model = torch.load(
            self.config.updated_base_model_path,
            map_location=self.device,
            weights_only=False
          )
        self.model.to(self.device)

    def train_valid_loader(self):
        image_size = self.config.params_image_size[:-1]  # (H, W)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.config.params_is_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(40),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        valid_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        full_dataset = datasets.ImageFolder(root=self.config.training_data)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size

        self.train_dataset, self.valid_dataset = random_split(full_dataset, [train_size, val_size])

        self.train_dataset.dataset.transform = train_transform
        self.valid_dataset.dataset.transform = valid_transform

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.params_batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.config.params_batch_size, shuffle=False)

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model, path)

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(self.config.params_epochs):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_corrects.float() / len(self.train_loader.dataset)

            print(f"Epoch {epoch+1}/{self.config.params_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # Validation (optional but included)
            self.model.eval()
            val_corrects = 0
            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == labels.data)

            val_acc = val_corrects.float() / len(self.valid_loader.dataset)
            print(f"Validation Acc: {val_acc:.4f}")

        # Save trained model
        self.save_model(self.config.trained_model_path, self.model)
