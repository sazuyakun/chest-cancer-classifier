import os
from pathlib import Path
import urllib.request as request
from zipfile import ZipFile
import torch
import torch.nn as nn
import torchvision.models as models
from chestCancerClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


    def get_base_model(self):
        # Load VGG16 Model
        self.model = models.vgg16(weights=models.VGG16_Weights[self.config.params_weights])
        if not self.config.params_include_top:
            self.model.classifier = nn.Identity()  # Remove the top layer

        self.model = self.model.to(self.device)
        self.save_model(path=self.config.base_model_path, model=self.model)



    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif freeze_all is not None and freeze_till > 0:
            for idx, child in enumerate(model.features.children()):
                if idx < freeze_till:
                    for param in child.parameters():
                        param.requires_grad = False

        full_model = nn.Sequential(
            model,
            nn.Flatten(),
            nn.Linear(25088, classes),
            nn.Softmax(dim=1)
        )

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, full_model.parameters()),
            lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        print(full_model)
        return full_model, optimizer, criterion


    def update_base_model(self):
        self.full_model, self.optimizer, self.criterion = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.full_model.to(self.device)
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)



    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model, path)
