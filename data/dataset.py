from torch.utils.data import Dataset


class MultiGraphDataset(Dataset):
    def __init__(self, tuples_list):
        self.data = tuples_list

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
