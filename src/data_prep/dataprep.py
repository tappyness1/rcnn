from torch.utils.data import Dataset
import torch
import yaml
import numpy as np
from typing import List # remove if python>=3.9

def convert_to_bbox_class(annotation: dict) -> torch.Tensor:
    # takes in the VOC dictionary and extracts
    # bbox as well as class name for each object in image
    
    return 

def extract_bndbox(annot: dict) -> np.array:
    """extracts bbox and class name from VOC annotation dataset

    Args:
        annot (dict): {annotation:{..., object: {name: str, bndbox: {xmin: str, ymin: str, xmax: str, ymax: str}} } }

    Returns:
        np.array: np.array(xmin, ymin, xmax, ymax, class)
    """
    # extract out the bounding boxes for each annotation passed in
    voc_class_file = open("data/voc_class_map.yaml")
    voc_class_map = yaml.load(voc_class_file, Loader=yaml.FullLoader)
    all_bboxes = []
    for obj in annot['annotation']['object']:
        class_name = voc_class_map[obj['name']]
        bbox = [obj['bndbox']['xmin'], obj['bndbox']['ymin'], obj['bndbox']['xmax'], obj['bndbox']['ymax']]
        bbox = [int(item) for item in bbox]
        bbox.append(class_name)
        all_bboxes.append(bbox)
    return np.array(all_bboxes)

def extract_all_bndbox(annots: List[dict]) -> List[np.array]:
    all_annots = []
    for annot in annots:
        all_annots.append(extract_bndbox(annot))
    return all_annots

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    import yaml
    from src.data_prep.dataset import get_load_data
    from torch.utils.data import DataLoader

    cfg_file = open("conf/cfg.yaml")
    cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    train, test = get_load_data(cfg)
    train_dataloader = DataLoader(train, batch_size=20, collate_fn=lambda batch: tuple(zip(*batch)))
    train_iter = iter(train_dataloader)
    img, class_bbox = next(iter(train_dataloader))
    print (extract_all_bndbox(class_bbox))

