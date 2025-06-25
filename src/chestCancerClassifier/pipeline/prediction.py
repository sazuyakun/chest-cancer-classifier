import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def predict(self):
        model_path = os.path.join("artifacts", "training", "model.pt")
        model = torch.load(model_path, map_location=self.device, weights_only=False)
        model.to(self.device)
        model.eval()

        image_size = (224, 224)  # (H, W)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_image = Image.open(self.filename).convert('RGB')
        test_image = transform(test_image).unsqueeze(0)  # Add batch dimension
        test_image = test_image.to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = model(test_image)
            _, predicted = torch.max(outputs, 1)
            result = predicted.cpu().numpy()[0]

        print(f"Prediction result: {result}")

        if result == 1:
            prediction = 'Normal'
        else:
            prediction = 'Adenocarcinoma Cancer'

        return [{"image": prediction}]
