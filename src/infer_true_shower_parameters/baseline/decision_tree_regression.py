import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from src.infer_true_shower_parameters import NUM_TRUE_SHOWER
import pandas as pd
import pickle
from src.common.regression_evaluate import mse, r2
from pathlib import Path

class DecisionTreeRegression(nn.Module):
    """
    Decision Tree Regressor that uses one tree for all output properties.
    """
    def __init__(self,  max_depth:int=5, ccp_alpha=0):
        super().__init__()
        self.tree = DecisionTreeRegressor(max_depth=max_depth, ccp_alpha=ccp_alpha)

    def set_params(self, max_depth=None, ccp_alpha=None):
        if max_depth:
            self.tree.set_params(max_depth=max_depth)
        if ccp_alpha != None:
            self.tree.set_params(ccp_alpha=ccp_alpha)

    def forward(self, X:torch.tensor):
        x_np = X.detach().cpu().numpy()
        pred = self.tree.predict(x_np)
        return torch.tensor(pred, dtype=torch.float32)
    
    def predict(self, X:np.array):
        return self.tree.predict(X)
    
    def fit(self, X:torch.tensor, y:torch.tensor):
        x_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        self.tree.fit(x_np, y_np)


def train(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    image_features: list[list[str]],
    additional_features: list[str],
    target_features: list[str],
    do_print=True,
    **kwargs
) -> DecisionTreeRegression:
    
    images_train = torch.concat([torch.tensor(train_data[feature].values, dtype=torch.float) for feature in image_features], dim=1)
    images_val = torch.concat([torch.tensor(validation_data[feature].values, dtype=torch.float) for feature in image_features], dim=1)
    add_train = torch.tensor(train_data[additional_features].values, dtype=torch.float)
    add_val = torch.tensor(validation_data[additional_features].values, dtype=torch.float)
    x_train = torch.cat([images_train, add_train], dim=1)
    x_val = torch.cat([images_val, add_val], dim=1)
    y_train = torch.tensor(train_data[target_features].values, dtype=torch.float)
    y_val = torch.tensor(validation_data[target_features].values, dtype=torch.float)

    model = DecisionTreeRegression(**kwargs)

    model.fit(x_train, y_train)

    if do_print:
        y_pred_train = model.predict(x_train)
        print("Train loss:", mse(y_pred_train, y_train.numpy()))
        print("Train R^2: ", r2(y_pred_train, y_train.numpy()))
        y_pred = model.predict(x_val)
        print("Validation loss:", mse(y_pred, y_val.numpy()))
        print("Validation R^2: ", r2(y_pred, y_val.numpy()))

    return model

def save(model, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model.tree, f)

def load(path: str):
    model = DecisionTreeRegression()
    with open(path, 'rb') as f:
        model.tree = pickle.load(f)
    return model

def evaluate(dir:Path, 
             name:str, 
             test_data:pd.DataFrame, 
             image_features:list[list[str]], 
             additional_features:list[str],
             target_features:list[str]) -> float:
    
    model = load(dir/f"{name}.pth")

    images_test = torch.concat([torch.tensor(test_data[feature].values, dtype=torch.float) for feature in image_features], dim=1)
    add_test = torch.tensor(test_data[additional_features].values, dtype=torch.float)
    x_test = torch.concat((images_test, add_test), dim=1)
    y_test = torch.tensor(test_data[target_features].values, dtype=torch.float)

    y_pred = model.predict(x_test)
    
    return r2(y_pred, y_test.numpy())