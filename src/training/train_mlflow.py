# from torchsummary import summary
import hydra
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.vit_model import ViT


def train(train_set, cfg, in_channels=3, num_classes=10):
    mlflow.set_tracking_uri(cfg["mlflow"]["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["mlflow_exp_name"])
    loss_function = nn.CrossEntropyLoss()

    # TODO: extract img size

    network = ViT(
        img_size=224, patch_dim=16, hidden_d=8, k_heads=2, num_classes=num_classes
    )

    network.train()

    optimizer = optim.SGD(
        network.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    network = network.to(device)

    train_dataloader = DataLoader(train_set, batch_size=20)
    with mlflow.start_run():
        mlflow.log_param("lr", cfg["train"]["lr"])
        for epoch in range(cfg["train"]["epochs"]):
            print(f"Epoch {epoch + 1}:")
            # for i in tqdm(range(X.shape[0])):
            with tqdm(train_dataloader) as tepoch:
                for imgs, labels in tepoch:
                    optimizer.zero_grad()
                    # only need to give [N x num_classes]. Loss function will do the rest for you. Probably an internal argmax
                    out = network(imgs.to(device))
                    loss = loss_function(out, labels.to(device))

                    mlflow.log_metric("loss", loss.item())

                    loss.backward()
                    optimizer.step()
                    tepoch.set_postfix(loss=loss.item())

    print("training done")
    # save model.state_dict() instead of model because of the flexibility
    torch.save(
        network.state_dict(),
        f"{cfg['save_model_path']}_epochs{cfg['train']['epochs']}_lr{cfg['train']['lr']}.pt",
    )

    # mlflow.log_artifact(cfg['save_model_path'])
    # artifact_uri = mlflow.get_artifact_uri()
    # mlflow.log_param("artifact_URI", artifact_uri)

    return network


if __name__ == "__main__":
    torch.manual_seed(42)

    from src.data_prep.dataset import get_load_data

    cfg = {
        "save_model_path": "model_weights/model_weights.pt",
        "epochs": 2,
        "show_model_summary": True,
        "train": {"lr": 0.001, "weight_decay": 5e-5},
    }
    train_set, test_set = get_load_data(root="../data", dataset="Flowers102")
    train(train_set=train_set, cfg=cfg, in_channels=3, num_classes=102)

    # cannot use FashionMNIST because size needs to be 224x224x3 at the very least
    # train_set, test_set = get_load_data(root = "../data", dataset = "FashionMNIST")
    # train(epochs = 1, train_set = train_set, in_channels = 1, num_classes = 10)
