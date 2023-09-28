import numpy as np


def box_iou_calc(boxes1: np.array, boxes2: np.array):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])
    

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min = 0, a_max = None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


if __name__ == "__main__":
    import numpy as np
    import torch
    from src.model.rcnn.rcnn import RCNN
    import yaml
    from src.data_prep.dataset import get_load_data
    from torch.utils.data import DataLoader
    from src.data_prep.dataprep import extract_all_bndbox

    np.random.seed(42)
    torch.manual_seed(42)

    # X = np.random.rand(5, 3, 227, 227).astype("float32")
    # X = torch.tensor(X)

    cfg_file = open("conf/cfg.yaml")
    cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    train, test = get_load_data(cfg)
    train_dataloader = DataLoader(train, batch_size=2, collate_fn=lambda batch: tuple(zip(*batch)))
    train_iter = iter(train_dataloader)
    img, class_bbox = next(iter(train_dataloader))
    model = RCNN(img_size = 227)
    output = model.forward(img)
    gt = extract_all_bndbox(class_bbox)
    # print (output)
    # print (gt)
    # print (output[0].detach().numpy()[:, :4].shape) # (100, 4)
    # print (gt[0][:,:4].shape) # (:, 4)
    print (box_iou_calc(output[0].detach().numpy()[:, :4], gt[0][:,:4]))
