import pandas as pd
import torch
import torch.nn as nn


def predict(model, img):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    return torch.argmax(model(img))
