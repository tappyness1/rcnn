import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def validation(model, val_set):
    """Simple validation workflow. Current implementation is for F1 score

    Args:
        model (_type_): _description_
        val_set (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()
    val_dataloader = DataLoader(val_set, batch_size=20)
    loss_function = nn.CrossEntropyLoss()
    y_true = []
    y_pred = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    with tqdm(val_dataloader) as tepoch:
        for imgs, labels in tepoch:
            y_true.extend(labels.numpy())

            with torch.no_grad():
                out = model(imgs.to(device))

            y_pred.extend(torch.argmax(out, 1).cpu().numpy())
            loss = loss_function(out, labels.to(device))

    cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
    # cm.to_csv("val_results/results.csv")
    return f1_score(y_true, y_pred, average="weighted")


if __name__ == "__main__":
    from src.data_prep.dataset import get_load_data

    _, val_set = get_load_data(root="../data", dataset="Flowers102")
    trained_model_path = "model_weights/model_weights.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(trained_model_path, map_location=torch.device(device))
    validation(model, val_set)
