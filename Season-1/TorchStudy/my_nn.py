import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, D_h, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_h)
        self.linear2 = torch.nn.Linear(D_h, D_out)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x