import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from infer_true_shower_parameters import NUM_TRUE_SHOWER
import pandas as pd
import pickle

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
    n_epochs=50,
    **kwargs
) -> DecisionTreeRegression:
    
    images_train = torch.concat([torch.tensor(train_data[feature].values, dtype=torch.float) for feature in image_features])
    images_val = torch.concat([torch.tensor(validation_data[feature].values, dtype=torch.float) for feature in image_features])
    add_train = torch.tensor(train_data[additional_features].values, dtype=torch.float)
    add_val = torch.tensor(validation_data[additional_features].values, dtype=torch.float)
    y_train = torch.tensor(train_data[target_features].values, dtype=torch.float)
    y_val = torch.tensor(validation_data[target_features].values, dtype=torch.float)

    model = DecisionTreeRegression(**kwargs)

    model.fit(images_train, y_train, epochs=n_epochs)

    return model

def save(model, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model.tree, f)

def load(path: str):
    model = DecisionTreeRegression()
    with open(path, 'rb') as f:
        model.tree = pickle.load(f)