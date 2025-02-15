import os

import numpy as np
import torch
from tqdm import tqdm

from src.parameters import Params
from src.dataloader import getDataloader
from src.model import MlpModel
from src.utils import compute_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("using", torch.cuda.get_device_name(0))
else:
    print("CUDA unavailable, using CPU")


params = Params()
params.batch_size = 8

test_data = np.load("data/test_data.npy")
test_label = np.load("data/test_label.npy")
test_loader = getDataloader(test_data, test_label, batch_size=params.batch_size)

model = MlpModel().to(device)
model.load_state_dict(torch.load("checkpoints/model.pth"))
model.eval()

test_acc = 0.0
with torch.no_grad():
    for data, label in tqdm(test_loader, desc="Test", unit="batch"):
        data = data.to(device)
        label = label.to(device)
        output = model(data).squeeze()
        test_acc += compute_accuracy(output, label)
    test_acc /= len(test_loader)

print("\nTest Accuracy {:.2f}%".format(100 * test_acc))
