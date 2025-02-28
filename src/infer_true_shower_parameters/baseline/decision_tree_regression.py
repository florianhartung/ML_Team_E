import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from infer_true_shower_parameters import NUM_TRUE_SHOWER

class DecisionTreeRegression(nn.Module):
    """
    Decision Tree Regressor that uses one tree for all output properties.
    """
    def __init__(self,  max_depth:int):
        super().__init__()
        self.tree = DecisionTreeRegressor(max_depth=max_depth)

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

class DecisionTreesRegression(nn.Module):
    """
    Decision Tree Regressor that uses one tree for each output property.
    """
    def __init__(self,  max_depth:int):
        super().__init__()
        self.trees = [DecisionTreeRegressor(max_depth=max_depth) for _ in range(NUM_TRUE_SHOWER)]

    def forward(self, X:torch.tensor):
        x_np = X.detach().cpu().numpy()
        pred = np.array([tree.predict(x_np) for tree in self.trees]).T
        return torch.tensor(pred, dtype=torch.float32)
    
    def predict(self, X:np.array):
        return np.array([tree.predict(X) for tree in self.trees]).T
    
    def fit(self, X:torch.tensor, y:torch.tensor):
        x_np = X.detach().cpu().numpy()
        ys_np = y.detach().cpu().numpy()
        for tree, y in zip(self.trees, ys_np.T):
            tree.fit(x_np, y)