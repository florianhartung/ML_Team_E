from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from src.common import visualizations

from src.common.HexaToParallelogram import HexaToParallelogram

# Decrease this for less memory consumption at the cost of performance for evaluation
VALIDATION_TEST_BATCH_SIZE = 1024


class ParticleClassificationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_features: list[list[str]],
        additional_features: list[str],
        class_feature: str,
        device: torch.device,
    ):
        self.images = [df[features].values for features in image_features]
        self.features = df[additional_features].values
        self.labels = df[[class_feature]].values
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        images = torch.stack(
            [torch.tensor(image[idx], dtype=torch.float) for image in self.images]
        )
        features = torch.tensor(self.features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return images, features, label


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 8, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, images):
        x = self.model(images)
        return x


class ParticleCNNClassifier(nn.Module):
    def __init__(self, num_additional_parameters):
        super().__init__()

        self.hex2par = HexaToParallelogram(2)
        self.cnn = CNN()
        self.fully_connected = nn.Sequential(
            nn.Linear(16 * 6 * 6 + num_additional_parameters, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, images, additional_parameters):
        x = self.hex2par(images)
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = torch.cat((x, additional_parameters), dim=1)
        x = self.fully_connected(x)

        return x


def train(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    image_features: list[list[str]],
    additional_features: list[str],
    class_feature: str,
    device: torch.device,
    pos_weight: float,
    epochs=10,
) -> ParticleCNNClassifier:
    train_loader = DataLoader(
        ParticleClassificationDataset(
            train_data, image_features, additional_features, class_feature, device
        ),
        batch_size=128,
        shuffle=True,
    )

    model = ParticleCNNClassifier(len(additional_features))
    model.to(device)

    training_loss_history = np.zeros(epochs)
    validation_loss_history = np.zeros(epochs)
    training_accuracy_history = np.zeros(epochs)
    validation_accuracy_history = np.zeros(epochs)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        training_loss = 0
        correct, total = 0, 0

        for images_batch, additional_features_batch, labels_batch in train_loader:
            optimizer.zero_grad()

            images_batch, additional_features_batch, labels_batch = (
                images_batch.to(device),
                additional_features_batch.to(device),
                labels_batch.to(device),
            )

            outputs = model(images_batch, additional_features_batch)

            loss = loss_function(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels_batch).sum().item()
            total += labels_batch.size(0)

        training_loss /= len(train_loader)
        training_accuracy = correct / total

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_data,
            image_features,
            additional_features,
            class_feature,
            device,
        )

        training_loss_history[epoch] = training_loss
        validation_loss_history[epoch] = validation_loss
        training_accuracy_history[epoch] = training_accuracy
        validation_accuracy_history[epoch] = validation_accuracy

        epoch_end = time.time()
        visualizations.print_training_progress(
            epoch,
            epochs,
            epoch_end - epoch_start,
            (training_loss, training_accuracy),
            (validation_loss, validation_accuracy),
        )

    visualizations.plot_training_validation(
        training_loss_history,
        training_accuracy_history,
        validation_loss_history,
        validation_accuracy_history,
    )

    return model


def evaluate(
    model: ParticleCNNClassifier,
    data: pd.DataFrame,
    image_features: list[list[str]],
    additional_features: list[str],
    class_feature: str,
    device: torch.device,
) -> (float, float):
    dataloader = DataLoader(
        ParticleClassificationDataset(
            data, image_features, additional_features, class_feature, device
        ),
        batch_size=VALIDATION_TEST_BATCH_SIZE,
    )

    model.eval()
    loss = 0
    correct, total = 0, 0

    loss_function = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, additional_features, labels in dataloader:
            images, additional_features, labels = (
                images.to(device),
                additional_features.to(device),
                labels.to(device),
            )

            outputs = model(images, additional_features)
            loss += loss_function(outputs, labels).item()
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return loss / len(dataloader), correct / total


def predict(
    model: ParticleCNNClassifier,
    data: pd.DataFrame,
    image_features: list[list[str]],
    additional_features: list[str],
    device: torch.device,
) -> pd.DataFrame:
    image_features = torch.swapaxes(
        torch.stack(
            [
                torch.tensor(data[image].values, dtype=torch.float)
                for image in image_features
            ]
        ),
        0,
        1,
    ).to(device)
    additional_features = torch.tensor(
        data[additional_features].values, dtype=torch.float
    ).to(device)
    return (
        torch.round(torch.sigmoid(model(image_features, additional_features)))
        .cpu()
        .detach()
    )

def save(model: ParticleCNNClassifier, path):
    torch.save(model.state_dict(), path)


def load(path, additional_features, device):
    model = ParticleCNNClassifier(len(additional_features))
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model
