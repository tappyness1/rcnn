import cv2
import numpy as np

def selective_search(image: np.array) -> np.ndarray:
    """Generates the coordinates of the proposed bounding boxes (region proposal)

    Args:
        image (np.array): An image that is in np.array. To convert from torch Tensor, use img.permute(1,2,0).numpy()
                          where img is the torch.Tensor object. The image must have been permuted from (3xHxW) to (HxWx3)

    Returns:
        np.ndarray: ndarray of N x (x,y,w,h) coordinates 
    """
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()

    # generate bounding boxes
    rects = ss.process()
    return rects

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    import yaml
    from src.data_prep.dataset import get_load_data
    from torch.utils.data import DataLoader
    import numpy as np

    cfg_file = open("conf/cfg.yaml")
    cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    train, test = get_load_data(cfg)
    train_dataloader = DataLoader(train, batch_size=20, collate_fn=lambda batch: tuple(zip(*batch)))
    train_iter = iter(train_dataloader)
    imgs, class_bboxes = next(iter(train_dataloader))
    # print (imgs[0].permute(1,2,0).numpy())
    print (type(selective_search(imgs[0].permute(1,2,0).numpy())))

