import numpy as np
import torch

def comptute_accuracy(output, label, gpu=True, grad=False):
    if gpu:
        output = output.cpu()
        label = label.cpu()
    if grad:
        output = output.detach()

    predict_label = np.where(output.numpy() < 0.5, 0, 1)
    label = label.numpy()
    return np.mean(predict_label == label)