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
    def __init__(self, n_trees:int, max_depth:int, n_jobs=32):
        super().__init__()
        self.forest = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, n_jobs=n_jobs)

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

class RandomForestsRegression(nn.Module):
    """
    Random Forest Regressor that uses one forest for each output property.
    """
    def __init__(self, n_trees:int, max_depth:int, n_jobs=32):
        super().__init__()
        self.forests = [RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, n_jobs=n_jobs) for _ in range(NUM_TRUE_SHOWER)]

    def forward(self, X:torch.tensor):
        x_np = X.detach().cpu().numpy()
        pred = np.array([forest.predict(x_np) for forest in self.forests]).T
        return torch.tensor(pred, dtype=torch.float32)
    
    def predict(self, X:np.array):
        return np.array([forest.predict(X) for forest in self.forests]).T
    
    def fit(self, X:torch.tensor, y:torch.tensor):
        x_np = X.detach().cpu().numpy()
        ys_np = y.detach().cpu().numpy()
        for forest, y in zip(self.forests, ys_np.T):
            forest.fit(x_np, y)