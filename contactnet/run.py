from contactnet.model.network import ContactNet
import torch


def train():
    model = ContactNet(3,3)
    inp = torch.randn(1,3,256,256)
    print(model(inp))