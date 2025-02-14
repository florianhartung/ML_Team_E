import torch
import torch.nn as nn
import torch.optim as optim
from infer_true_shower_parameters import NUM_TRUE_SHOWER

class LinearRegression(nn.Module):
    def __init__(self, input_size:int):
        super().__init__()
        self.linear = nn.Linear(input_size, NUM_TRUE_SHOWER)

    def forward(self, x):
        return self.linear(x)
    
    def predict(self, X):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(X).numpy()
        
    def test(self):
        print("test")

    def fit(self, X, y, lr=0.01, epochs=1000):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        assert not torch.isnan(X).any(), "X contains NaN values"
        assert not torch.isnan(y).any(), "y contains NaN values"

        criterion = nn.MSELoss()  
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(X)
            loss = criterion(y_pred, y)
            loss.backward()
            # for name, param in self.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} gradient: {param.grad.abs().mean()}")

            optimizer.step()