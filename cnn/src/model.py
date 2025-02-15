import torch


class CnnModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(CnnModel, self).__init__()

        self.cnn_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.mlp_layer = torch.nn.Sequential(
            torch.nn.Linear(7 * 7 * 32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = torch.flatten(x, start_dim=1)  # flatten image for mlp layer
        x = self.mlp_layer(x)
        return x
