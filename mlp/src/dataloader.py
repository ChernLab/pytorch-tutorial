import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def getDataloaders(data, label, split_list=None, batch_size=1, shuffle=False):

    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).float()
    dataset = TensorDataset(data, label)

    if split_list is not None:
        dataset1, dataset2 = random_split(dataset, split_list)
        loader1 = DataLoader(dataset=dataset1, batch_size=batch_size, shuffle=shuffle)
        loader2 = DataLoader(dataset=dataset2, batch_size=batch_size, shuffle=shuffle)
        return loader1, loader2
    else:
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        return loader