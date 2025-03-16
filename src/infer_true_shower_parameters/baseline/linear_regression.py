import torch
import torch.nn as nn
import torch.optim as optim
from infer_true_shower_parameters import NUM_TRUE_SHOWER
import pandas as pd

class LinearRegression(nn.Module):
    def __init__(self, input_size:int):
        super().__init__()
        self.linear = nn.Linear(input_size, NUM_TRUE_SHOWER)

    def forward(self, x):
        return self.linear(x)
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(X).numpy()
        
    def test(self):
        print("test")

    def fit(self, X, y, lr=0.01, epochs=100):
        assert not torch.isnan(X).any(), "X contains NaN values"
        assert not torch.isnan(y).any(), "y contains NaN values"

        criterion = nn.MSELoss()  
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(X)
            loss = criterion(y_pred, y)
            loss.backward()
            # print(loss.item())

            optimizer.step()

def train(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    image_features: list[list[str]],
    additional_features: list[str],
    target_features: list[str],
    device: torch.device,
    n_epochs=50
) -> LinearRegression:
    
    images_train = torch.concat([torch.tensor(train_data[feature].values, dtype=torch.float) for feature in image_features])
    images_val = torch.concat([torch.tensor(validation_data[feature].values, dtype=torch.float) for feature in image_features])
    add_train = torch.tensor(train_data[additional_features].values, dtype=torch.float)
    add_val = torch.tensor(validation_data[additional_features].values, dtype=torch.float)
    y_train = torch.tensor(train_data[target_features].values, dtype=torch.float)
    y_val = torch.tensor(validation_data[target_features].values, dtype=torch.float)

    model = LinearRegression(len(image_features) + len(additional_features)).to(device)

    model.fit(images_train, y_train, epochs=n_epochs)

    return model

def save(model, path):
    torch.save(model.state_dict(), path)

def load(path):
    return torch.load(path)