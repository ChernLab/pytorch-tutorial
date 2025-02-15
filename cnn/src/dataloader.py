import os
import struct
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


# read MNIST
def read_mnist(path, load_type="train"):
    def load_mnist_image(filename):
        with open(filename, "rb") as f:
            _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            images = (
                np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols) / 255.0
            )  # normalize
            return images[
                :, np.newaxis, :, :
            ]  # add grayscale dimension (num_images, 1, height, width)

    def load_mnist_label(filename):
        with open(filename, "rb") as f:
            _, num_labels = struct.unpack(">II", f.read(8))
            return np.fromfile(f, dtype=np.uint8)

    if load_type == "train":
        image = load_mnist_image(os.path.join(path, "train-images.idx3-ubyte"))
        label = load_mnist_label(os.path.join(path, "train-labels.idx1-ubyte"))
    elif load_type == "test":
        image = load_mnist_image(os.path.join(path, "t10k-images.idx3-ubyte"))
        label = load_mnist_label(os.path.join(path, "t10k-labels.idx1-ubyte"))
    else:
        print("load type should be either train or test")
        exit(1)
    return image, label


# dataloader
def getDataloader(image, label, split_list=None, batch_size=1, shuffle=False):
    image = torch.from_numpy(image).float()
    label = torch.from_numpy(label).long()
    dataset = TensorDataset(image, label)

    if split_list is not None:
        dataset1, dataset2 = random_split(dataset, split_list)
        loader1 = DataLoader(dataset=dataset1, batch_size=batch_size, shuffle=shuffle)
        loader2 = DataLoader(dataset=dataset2, batch_size=batch_size, shuffle=shuffle)
        return loader1, loader2
    else:
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
