import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from infer_true_shower_parameters import NUM_TRUE_SHOWER

class RandomForestRegression(nn.Module):
    """
    Random Forest Regressor that uses one forrest for all output properties.
    """
    def __init__(self, n_trees:int=100, max_depth:int=10, ccp_alpha=0, n_jobs=32):
        super().__init__()
        self.forest = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, n_jobs=n_jobs, ccp_alpha=ccp_alpha)

    def set_params(self, max_depth=None, n_trees=None, ccp_alpha=None):
        if max_depth:
            self.forest.set_params(max_depth=max_depth)
        if n_trees:
            self.forest.set_params(n_estimators=n_trees)
        if ccp_alpha != None:
            self.forest.set_params(ccp_alpha=ccp_alpha)

    def forward(self, X:torch.tensor):
        x_np = X.detach().cpu().numpy()
        pred = self.forest.predict(x_np)
        return torch.tensor(pred, dtype=torch.float32)
    
    def predict(self, X:np.array):
        return self.forest.predict(X)
    
    def fit(self, X:torch.tensor, y:torch.tensor):
        x_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        self.forest.fit(x_np, y_np)
