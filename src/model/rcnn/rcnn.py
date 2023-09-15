from src.model.rcnn.selective_search import selective_search
from src.model.rcnn.alexnet import AlexNet
import torch.nn as nn
import torch
import numpy as np
from torchvision.transforms import Resize

class RCNN(nn.Module):

    def __init__(self, img_size = 227, num_classes = 10):
        super(RCNN, self).__init__()
        self.backbone = AlexNet(img_size, num_classes)
        self.resize = Resize((227, 227))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor: 
        outputs = []
        for inp in input:
            coords = selective_search(inp.permute(1,2,0).numpy())
            output = []
            for coord in coords:
                img = inp[:, coord[1]: coord[1] + coord[3], coord[0]: coord[0] + coord[2]]
                img = self.resize(img)
                out = torch.argmax(self.backbone(img.reshape(1,3,227,227)))
                output.append(np.append(coord,out))
            outputs.append(output)
        return outputs

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