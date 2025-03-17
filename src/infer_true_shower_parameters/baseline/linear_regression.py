import torch
import torch.nn as nn
import torch.optim as optim
from src.infer_true_shower_parameters import NUM_TRUE_SHOWER
import pandas as pd
from src.common.regression_evaluate import mse, r2
from pathlib import Path

class LinearRegression(nn.Module):
    def __init__(self, input_size:int):
        super().__init__()
        self.linear = nn.Linear(input_size, NUM_TRUE_SHOWER)

    def forward(self, x):
        return self.linear(x)
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(X).cpu().numpy()
        
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
    n_epochs=50,
    do_print=True
) -> LinearRegression:
    
    images_train = torch.concat([torch.tensor(train_data[feature].values, dtype=torch.float) for feature in image_features], dim=1)
    images_val = torch.concat([torch.tensor(validation_data[feature].values, dtype=torch.float) for feature in image_features], dim=1)
    add_train = torch.tensor(train_data[additional_features].values, dtype=torch.float)
    add_val = torch.tensor(validation_data[additional_features].values, dtype=torch.float)
    x_train = (torch.concat((images_train, add_train), dim=1) if len(additional_features) > 0 else images_train).to(device)
    x_val = (torch.concat((images_val, add_val), dim=1) if len(additional_features) > 0 else images_val).to(device)
    y_train = torch.tensor(train_data[target_features].values, dtype=torch.float).to(device)
    y_val = torch.tensor(validation_data[target_features].values, dtype=torch.float).to(device)

    model = LinearRegression(sum([len(feature) for feature in image_features]) + len(additional_features)).to(device)

    model.fit(x_train, y_train, epochs=n_epochs)

    if do_print:
        y_pred_train = model.predict(x_train)
        print("Train loss:", mse(y_pred_train, y_train.cpu().numpy()))
        print("Train R^2: ", r2(y_pred_train, y_train.cpu().numpy()))
        y_pred = model.predict(x_val)
        print("Validation loss:", mse(y_pred, y_val.cpu().numpy()))
        print("Validation R^2: ", r2(y_pred, y_val.cpu().numpy()))

    return model

def save(model, path):
    torch.save(model.state_dict(), path)

def load(path, **kwargs):
    model = LinearRegression(**kwargs)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

def evaluate(dir:Path, 
             name:str, 
             test_data:pd.DataFrame, 
             image_features:list[list[str]], 
             additional_features:list[str],
             target_features:list[str]) -> float:
    model = load(dir/f"{name}.pth", input_size=sum([len(feature) for feature in image_features]) + len(additional_features))
    images_test = torch.concat([torch.tensor(test_data[feature].values, dtype=torch.float) for feature in image_features], dim=1)
    add_test = torch.tensor(test_data[additional_features].values, dtype=torch.float)
    x_test = (torch.concat((images_test, add_test), dim=1) if len(additional_features) > 0 else images_test)
    y_test = torch.tensor(test_data[target_features].values, dtype=torch.float)
    y_pred = model.predict(x_test)
    
    return r2(y_pred, y_test.numpy())