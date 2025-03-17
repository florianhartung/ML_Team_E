import time

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader

from src.common.batch import BatchDataset
from src.common import visualizations

def create_dataset(data: pd.DataFrame, features: list[str], class_feature: str, device):
    return BatchDataset(
        device, data[features], data[[class_feature]]
    )

def evaluate(model, device, dataset: BatchDataset):
    dataloader = DataLoader(dataset, batch_size=1024)
    loss_function = nn.BCEWithLogitsLoss()

    model.eval()
    with torch.no_grad():
        test_batch_losses = []
        test_batch_accuracies = []

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            test_output = model(x)
            test_batch_loss = loss_function(test_output, y)
            test_batch_losses.append(test_batch_loss.item())
            test_class_predictions = (torch.round(torch.sigmoid(test_output)) == y).float()
            test_batch_accuracies.append(test_class_predictions.cpu().mean())

        return (np.mean(test_batch_losses), np.mean(test_batch_accuracies))

class MlpClassifier(nn.Module):
    def __init__(self, num_features: int, dropout: float):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features, 256),
            #nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(256, 128),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(128, 64),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

def train(
        train_data: pd.DataFrame, 
        validation_data: pd.DataFrame, 
        features: list[str],
        class_feature: str,
        pos_weight: float,
        device: torch.device,
        epochs: int = 11,
) -> MlpClassifier:
    loss_history = np.zeros(shape=epochs)
    accuracy_history = np.zeros(shape=epochs)
    validation_loss_history = np.zeros(shape=epochs)
    validation_accuracy_history = np.zeros(shape=epochs)
    train_dataloader = DataLoader(create_dataset(train_data, features, class_feature, device), batch_size=64, shuffle=True)
    validation_dataset = create_dataset(validation_data, features, class_feature, device)

    model = MlpClassifier(len(features), 0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))#pos_weight = torch.tensor([7.8]).to(device))

    for epoch in range(epochs):
        start = time.time()

        model.train()

        batch_losses = []
        batch_accuracies = []

        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
        
            batch_loss = loss(output, y)
            batch_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            batch_losses.append(batch_loss.item())
            class_predictions = (torch.round((torch.sigmoid(output))) == y).float()
            batch_accuracies.append(class_predictions.cpu().mean())

        epoch_loss = np.mean(batch_losses)
        epoch_accuracy = np.mean(batch_accuracies)
        accuracy_history[epoch] = epoch_accuracy
        loss_history[epoch] = epoch_loss

        validation_loss, validation_accuracy = evaluate(model, device, validation_dataset)
 
        validation_loss_history[epoch] = validation_loss
        validation_accuracy_history[epoch] = validation_accuracy

        end = time.time()

        visualizations.print_training_progress(
            epoch,
            epochs,
            end - start,
            (epoch_loss, epoch_accuracy),
            (validation_loss, validation_accuracy)
        )

    visualizations.plot_training_validation(
        loss_history,
        accuracy_history,
        validation_loss_history,
        validation_accuracy_history
    )

    return model

def save(model, path):
    torch.save(model.state_dict(), path)

def load(path, num_features, device):
    return MlpClassifier(num_features, 0.2).to(device)