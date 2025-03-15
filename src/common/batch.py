import torch

class BatchDataset(torch.utils.data.Dataset):
    """
    Usage:
    >> from torch.utils.data import DataLoader
    >> dataset = BatchDataset(X_train, y_train)
    >> loader = DataLoader(dataset, batch_size=64) # This should be done in each epoch
    >> for x, y in loader: # x and y are batches
    >>      ...
    """
    def __init__(self, X_train, y_train):
        super(BatchDataset, self).__init__()

        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return self.y_train.shape[0]

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]