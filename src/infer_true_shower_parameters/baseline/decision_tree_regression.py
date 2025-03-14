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
    def __init__(self,  max_depth:int=10, ccp_alpha=0):
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