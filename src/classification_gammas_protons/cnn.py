from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time

from src.common.HexaToParallelogram import HexaToParallelogram


class ParticleClassificationDataset(Dataset):
    def __init__(
        self,
        df,
        image_features: list[list[str]],
        additional_features,
        class_feature,
        device,
    ):
        self.images = [df[features].values for features in image_features]
        self.features = df[additional_features].values
        self.labels = df[[class_feature]].values
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        images = torch.stack(
            [
                torch.tensor(image[idx], dtype=torch.float).to(self.device)
                for image in self.images
            ]
        ).to(self.device)
        features = torch.tensor(self.features[idx], dtype=torch.float).to(self.device)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).to(self.device)
        return images, features, label


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 8, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 5),
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
    epochs=10,
) -> ParticleCNNClassifier:
    batch_size = 32

    # Build torch datasets and dataloaders (this does not copy the data)
    make_dataset = lambda data: ParticleClassificationDataset(
        data, image_features, additional_features, class_feature, device
    )
    train_loader = DataLoader(
        make_dataset(train_data), batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        make_dataset(validation_data), batch_size=batch_size, shuffle=False
    )

    model = ParticleCNNClassifier(len(additional_features))
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-2)

    for epoch in range(epochs):

        epoch_start = time.time()
        model.train()
        epoch_accum_loss = 0
        correct, total = 0, 0

        for images, additional_features, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images, additional_features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_accum_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_accum_loss /= len(train_loader)
        epoch_accuracy = correct / total

        val_loss, val_acc = evaluate_model(model, validation_loader, criterion, device)

        epoch_end = time.time()
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_accum_loss:.8f} Acc: {epoch_accuracy:.8f} | Val Loss: {val_loss:.8f} Val Acc: {val_acc:.8f} | Epoch took {epoch_end - epoch_start:.2f}s"
        )

    return model


def evaluate_model(
    model: ParticleCNNClassifier, dataloader, criterion, device
) -> (float, float):
    model.eval()
    loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for images, additional_features, labels in dataloader:

            outputs = model(images, additional_features)
            loss += criterion(outputs, labels).item()
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return loss / len(dataloader), correct / total


def save(model: ParticleCNNClassifier, path):
    torch.save(model.state_dict(), path)

def load(path, additional_features):
    model = ParticleCNNClassifier(len(additional_features))
    model.load_state_dict(torch.load(path))
    return model
