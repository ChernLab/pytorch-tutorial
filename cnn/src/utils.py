import numpy as np
import torch


def compute_accuracy(output, label, gpu=True, grad=False):
    if gpu:
        output = output.cpu()
        label = label.cpu()
    if grad:
        output = output.detach()

    predict_label = torch.argmax(output, dim=1)
    predict_label = predict_label.numpy()
    label = label.numpy()
    return np.mean(predict_label == label)
