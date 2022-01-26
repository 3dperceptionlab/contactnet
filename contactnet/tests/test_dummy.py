import pytest
import torch
from contactnet.model.network import ContactNet

model = ContactNet(in_channels=3, out_channels=3)

def test_random_input():
    inp = torch.randn(1, 3, 256, 256)
    out = model(inp)

    assert out.shape == (1, 3, 256, 256)

