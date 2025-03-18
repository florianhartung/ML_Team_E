import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common.HexaToParallelogram import HexaToParallelogram
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from src.common.batch import BatchDataset

class ParticleMLPRegressor(nn.Module):
    def __init__(self, input_size, fc_layers=[128, 64, 32, 8], dropout_rates=[0, 0, 0]):
        super().__init__()
        
        assert len(fc_layers) >= 2, "At least two fully connected layers are required"
        assert len(dropout_rates) == len(fc_layers) - 1, "dropout_rates must have length equal to len(fc_layers) - 1"
        
        fc_layers = [input_size] + fc_layers
        
        layers = []
        for i in range(len(fc_layers) - 1):
            layers.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(fc_layers[i + 1]))
            if i < len(fc_layers) - 2: 
                layers.append(nn.Dropout(dropout_rates[i]))
        
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.fc(x)
        return x

def r_sq(pred, actual):
    return 1 - torch.sum((pred - actual) ** 2) / torch.sum((actual - torch.mean(actual)) ** 2)

def train(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    image_features: list[list[str]],
    additional_features: list[str],
    target_features: list[str],
    device: torch.device,
    n_epochs=20,
    plot=False,
    weight_decay=1e-4,
    learning_rate=1e-4,
    **kwargs
) -> ParticleMLPRegressor:

    x_train_df = pd.concat([train_data[f] for f in image_features] + [train_data[additional_features]], axis=1)
    x_val_df = pd.concat([validation_data[f] for f in image_features] + [validation_data[additional_features]], axis=1)
    y_train_df = train_data[target_features]
    y_val_df = validation_data[target_features]

    dataset = BatchDataset(device, x_train_df, y_train_df)
    test_dataset = BatchDataset(device, x_val_df, y_val_df)

    train_r2s, val_r2s = [], []
    train_loss, val_loss = [], []

    model = ParticleMLPRegressor(input_size=len(x_train_df.columns), **kwargs).to(device)
    optimizer = optim.Adam(model.parameters(recurse=True), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        print(f"epoch {epoch + 1}")

        model.train()

        running_loss = 0.0
        running_r2 = 0.0

        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        for x, y in loader: # x and y are batches
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_r2 += r_sq(outputs, y).item()

        if plot:
         
            train_r2s.append(running_r2 / len(loader))
            train_loss.append(running_loss / len(loader))

            model.eval()
            with torch.no_grad():
                val_loader = DataLoader(test_dataset, batch_size=64)
                running_loss = 0.0
                running_r2 = 0.0
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)

                    outputs = model(x)

                    loss = criterion(outputs, y)

                    running_loss += loss.item()
                    running_r2 += r_sq(outputs, y).item()
                
                val_r2s.append(running_r2 / len(val_loader))
                val_loss.append(running_loss / len(val_loader))

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10,5))

        axs[0].plot(train_r2s, label="train r2")
        axs[0].plot(val_r2s, label="val r2")
        axs[0].set_ylim(0, 1)
        axs[0].set_title('R^2 Score')
        axs[0].legend()

        axs[1].plot(train_loss, label="train loss")
        axs[1].plot(val_loss, label="val loss")
        axs[1].set_title('Loss')
        axs[1].legend()

        plt.tight_layout()
        plt.show()
        print(val_r2s[-1])

    return model

def save(model, path):
    torch.save(model.state_dict(), path)

def load(path, **kwargs):
    model = ParticleMLPRegressor(**kwargs)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

def evaluate(dir:Path, 
             name:str, 
             test_data:pd.DataFrame, 
             image_features:list[list[str]], 
             additional_features:list[str],
             target_features:list[str],
             device: torch.device
             ) -> float:
    
    model = load(dir/f"{name}.pth", input_size=len(image_features) + len(additional_features)).to(device)

    x_test = torch.concat([torch.tensor(test_data[feature].values, dtype=torch.float) for feature in image_features]
                          + [torch.tensor(test_data[additional_features].values, dtype=torch.float)], dim=1).to(device)
    y_test = torch.tensor(test_data[target_features].values, dtype=torch.float).to(device)

    model.eval()
    y_pred = model(x_test)

    return r_sq(y_pred, y_test).cpu().item()