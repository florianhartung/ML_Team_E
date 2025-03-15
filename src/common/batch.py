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
    def __init__(self, *datasets):
        super(BatchDataset, self).__init__()

        self.datasets = [*datasets]
        assert all([dataset.shape[0] == datasets[0].shape[0] for dataset in datasets])

    def __len__(self):
        return self.datasets[0].shape[0]

    def __getitem__(self, index):
        return [dataset[index] for dataset in self.datasets]