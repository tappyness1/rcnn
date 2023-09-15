import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor

# import matplotlib.pyplot as plt

def get_load_data(cfg: dict):
    root = cfg['dataset']['root']
    dataset = cfg['dataset']['dataset']
    download = cfg['dataset']['download']

    if dataset == "FashionMNIST":
        training_data = datasets.FashionMNIST(
            root=root, train=True, download=download, transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root=root, train=False, download=download, transform=ToTensor()
        )

    elif dataset == "Flowers102":
        training_data = datasets.Flowers102(
            root=root,
            split="test",
            download=download,
            transform=Compose([Resize((224, 224)), ToTensor()]),
        )

        test_data = datasets.Flowers102(
            root=root,
            split="train",
            download=download,
            transform=Compose([Resize((224, 224)), ToTensor()]),
        )

    elif dataset == "OxfordIIITPet":
        training_data = datasets.OxfordIIITPet(
            root=root,
            split ="trainval",
            target_types = "segmentation",
            download=download,
            transform=Compose([Resize((227, 227)),ToTensor()]),
            target_transform = Compose([Resize((227, 227)),ToTensor()])
        )

        test_data = datasets.OxfordIIITPet(
            root=root,
            split ="test",
            target_types = "segmentation",
            download=download,
            transform=Compose([Resize((227, 227)), ToTensor()]),
            target_transform = Compose([Resize((227, 227)),ToTensor()])
        )

    elif dataset == "VOCDetection":
        training_data = datasets.VOCDetection(
            root=root,
            image_set ="train",
            download=download,
            transform=Compose([Resize((227, 227)),ToTensor()]),
            # target_transform = Compose([Resize((227, 227)),ToTensor()])
        )

        test_data = datasets.VOCDetection(
            root=root,
            image_set ="val",
            download=download,
            transform=Compose([Resize((227, 227)), ToTensor()]),
            # target_transform = Compose([Resize((227, 227)),ToTensor()])
        )

    else:
        raise Exception("Enter a valid dataset string - OxfordIIITPet, Flowers102, FashionMNIST")

    return training_data, test_data


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    import yaml
    cfg_file = open("conf/cfg.yaml")
    cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    train, test = get_load_data(cfg)
    # img, label = train[1]
    # plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()
    # custom collate function to prevent torch.stack
    train_dataloader = DataLoader(train, batch_size=20, collate_fn=lambda batch: tuple(zip(*batch)))
    print (next(iter(train_dataloader)))
