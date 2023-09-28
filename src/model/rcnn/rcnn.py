from src.model.rcnn.selective_search import selective_search
from src.model.rcnn.alexnet import AlexNet
import torch.nn as nn
import torch
import numpy as np
from torchvision.transforms import Resize

class RCNN(nn.Module):

    def __init__(self, img_size = 227, num_classes = 10, max_dets = 100):
        super(RCNN, self).__init__()
        self.backbone = AlexNet(img_size, num_classes)
        self.resize = Resize((227, 227))
        self.max_dets = max_dets
    
    def forward(self, input: torch.Tensor) -> torch.Tensor: 
        outputs = []
        for inp in input:
            coords = selective_search(inp.permute(1,2,0).numpy())
            num_dets = min(self.max_dets,len(coords))
            coords = coords[:num_dets]
            output = []
            imgs = []
            coords_pvoc = [] # to change to pascal voc format because iou calculations
            for coord in coords:
                coord_pvoc = [coord[1], coord[0], coord[1] + coord[3], coord[0] + coord[2]]
                img = inp[:, coord[1]: coord[1] + coord[3], coord[0]: coord[0] + coord[2]]
                img = self.resize(img)
                coords_pvoc.append(coord_pvoc)
                imgs.append(img.reshape(3,227,227))
            imgs = torch.stack(imgs)
            out = torch.argmax(self.backbone(imgs), dim = 1).reshape((num_dets,1))
            conf = torch.max(self.backbone(imgs), dim = 1).values.reshape((num_dets,1))
            coords_pvoc = torch.Tensor(coords_pvoc)
            output = torch.cat([coords_pvoc, out, conf], dim = 1)
            outputs.append(output)
        return torch.stack(outputs)

if __name__ == "__main__":
    import numpy as np
    import torch
    # from torchsummary import summary

    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.rand(5, 3, 227, 227).astype("float32")
    X = torch.tensor(X)

    model = RCNN(img_size = 227)

    # summary(model, (3, 227, 227))
    # print(model.forward(X).shape)
    print(model.forward(X))