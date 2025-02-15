import os

import numpy as np
import torch
from tqdm import tqdm

from src.parameters import Params
from src.dataloader import getDataloaders
from src.model import MlpModel
from src.logger import Logger
from src.utils import compute_accuracy
from src.visualize import plot_loss_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("using", torch.cuda.get_device_name(0))
else:
    print("CUDA unavailable, using CPU")


# hyperparameters
params = Params()
params.num_epochs = 1000
params.early_stop = 10
params.batch_size = 8
params.learn_rate = 0.001
params.print_options()

# dataloader
train_data = np.load("data/train_data.npy")
train_label = np.load("data/train_label.npy")
train_loader, valid_loader = getDataloaders(
    train_data,
    train_label,
    split_list=[600, 200],
    batch_size=params.batch_size,
    shuffle=True,
)

# define model, loss, and optimizer
model = MlpModel().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params.learn_rate)

# train model
early_step = 0
min_valid_loss = 1e10
logger = Logger(params.num_epochs)

for epoch in range(params.num_epochs):
    train_acc, valid_acc = 0.0, 0.0
    train_loss, valid_loss = 0.0, 0.0
    # train
    for data, label in tqdm(
        train_loader,
        desc="Train [{:2d}/{}]".format(epoch + 1, params.num_epochs),
        unit="batch",
    ):
        # use CUDA if available
        data = data.to(device)
        label = label.to(device)
        # compute output and loss
        output = model(data).squeeze()
        loss = criterion(output, label)
        # update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update train loss and accuracy
        train_loss += loss.item()
        train_acc += compute_accuracy(output, label, grad=True)
    # validation
    model.eval()
    with torch.no_grad():
        for data, label in tqdm(
            valid_loader,
            desc="Valid [{:2d}/{}]".format(epoch + 1, params.num_epochs),
            unit="batch",
        ):
            # use CUDA if available
            data = data.to(device)
            label = label.to(device)
            # compute output and loss
            output = model(data).squeeze()
            loss = criterion(output, label)
            # update validation loss and accuracy
            valid_loss += loss.item()
            valid_acc += compute_accuracy(output, label)

    logger.write_loss(train_loss / len(train_loader), valid_loss / len(valid_loader))
    logger.write_acc(train_acc / len(train_loader), valid_acc / len(valid_loader))

    # early stop
    if min_valid_loss > valid_loss:
        early_step = 0
        min_valid_loss = valid_loss
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/model.pth")
    else:
        early_step += 1
        if early_step >= params.early_stop:
            logger.early_stop = epoch - params.early_stop
            print("No improvements for {} consecutive epochs".format(params.early_stop))
            break

# plot
os.makedirs("docs", exist_ok=True)
plot_loss_accuracy(logger, file_path="docs/training_loss_accuracy.png")
