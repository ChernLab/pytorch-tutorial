import torch

class MlpModel(torch.nn.Module):
    def __init__(self):
        super(MlpModel, self).__init__()

        self.layer = torch.nn.Sequential(
            torch.nn.Linear(2, 32), 
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16), 
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x): 

        return self.layer(x)