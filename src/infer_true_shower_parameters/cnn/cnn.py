import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common.HexaToParallelogram import HexaToParallelogram
import pandas as pd
from src.common.batch import BatchDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, in_channels, conv_channels, kernel_size, pooling_types):
        super().__init__()
        
        assert len(conv_channels) == len(pooling_types), "conv_channels and pooling_types must have the same length"
        
        layers = []
        input_channels = in_channels
        
        for i, (out_channels, pool_type) in enumerate(zip(conv_channels, pooling_types)):
            layers.append(nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))
            
            if pool_type == "max":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif pool_type == "strided":
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2))
            elif pool_type == "mean":
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                raise ValueError(f"Unknown pooling type: {pool_type}")
            
            input_channels = out_channels
        
        layers.append(nn.Flatten())
        self.convolutions = nn.Sequential(*layers)
    
    def forward(self, images):
        return self.convolutions(images)

class ParticleCNNRegressor(nn.Module):
    def __init__(self, num_additional_parameters, in_channels=2, conv_channels=[16, 32, 64], kernel_size=3, 
                 pooling_types=["max", "max", "max"], fc_layers=[128, 64, 8], dropout_rates=[0.0, 0.0]):
        super().__init__()
        
        assert len(fc_layers) >= 2, "At least two fully connected layers are required"
        assert len(dropout_rates) == len(fc_layers) - 1, "dropout_rates must have length equal to len(fc_layers) - 1"
        
        self.hex2par = HexaToParallelogram(in_channels)
        self.cnn = CNN(in_channels, conv_channels, kernel_size, pooling_types)
        
        # Compute flattened output size after CNN
        dummy_input = torch.zeros(1, in_channels, 39, 39)
        cnn_output_size = self.cnn(dummy_input).shape[1]
        
        fc_layers = [cnn_output_size + num_additional_parameters] + fc_layers
        
        layers = []
        for i in range(len(fc_layers) - 1):
            layers.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(fc_layers[i + 1]))
            if i < len(fc_layers) - 2:  # Apply dropout except for the last layer
                layers.append(nn.Dropout(dropout_rates[i]))
        
        self.fc = nn.Sequential(*layers)
    
    def forward(self, images, additional_parameters):
        x = self.hex2par(images)
        x = self.cnn(x)
        x = torch.cat((x, additional_parameters), 1)
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
    **kwargs
) -> ParticleCNNRegressor:

    images_train = torch.stack([torch.tensor(train_data[feature].values, dtype=torch.float) for feature in image_features], 1)
    images_val = torch.stack([torch.tensor(validation_data[feature].values, dtype=torch.float) for feature in image_features], 1)
    add_train = torch.tensor(train_data[additional_features].values, dtype=torch.float)
    add_val = torch.tensor(validation_data[additional_features].values, dtype=torch.float)
    y_train = torch.tensor(train_data[target_features].values, dtype=torch.float)
    y_val = torch.tensor(validation_data[target_features].values, dtype=torch.float)

    dataset = BatchDataset(images_train, add_train, y_train)
    test_dataset = BatchDataset(images_val, add_val, y_val)

    train_r2s, val_r2s = [], []
    train_loss, val_loss = [], []

    model = ParticleCNNRegressor(num_additional_parameters=len(additional_features), **kwargs).to(device)
    optimizer = optim.Adam(model.parameters(recurse=True), lr=0.001, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        print(f"epoch {epoch + 1}")

        model.train()

        running_loss = 0.0
        running_r2 = 0.0
        i = 0

        loader = DataLoader(dataset, batch_size=64, shuffle=True) # This should be done in each epoch
        for image, addition, y in loader: # x and y are batches
            img = image.to(device)
            add = addition.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            outputs = model(img, add)

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
                for image, addition, y in val_loader:
                    img = image.to(device)
                    add = addition.to(device)
                    y = y.to(device)

                    outputs = model(img, add)

                    loss = criterion(outputs, y)

                    running_loss += loss.item()
                    running_r2 += r_sq(outputs, y).item()
                
                val_r2s.append(running_r2 / len(val_loader))
                val_loss.append(running_loss / len(val_loader))

    if plot:
        plt.plot(train_r2s, label="train r2")
        plt.plot(val_r2s, label="val r2")
        plt.legend()
        plt.show()
        plt.plot(train_loss, label="train loss")
        plt.plot(val_loss, label="val loss")
        plt.legend()
        plt.show()

    return model

def save(model, path):
    torch.save(model.state_dict(), path)

def load(path, **kwargs):
    # model = ParticleCNNRegressor(**kwargs)
    # model.load_state_dict(torch.load(path))
    return torch.load(path)