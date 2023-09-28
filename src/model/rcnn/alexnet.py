import numpy as np
import torch
import torch.nn as nn
from torch.nn import (
    Conv2d,
    Linear,
    MaxPool2d,
    ReLU,
    Softmax,
    LocalResponseNorm,
    Dropout,
    Sigmoid
)

class AlexNet(nn.Module):
    def __init__(self, img_size=227, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer_1 = Conv2d(
                            in_channels=3,
                            out_channels=96,
                            kernel_size=11,
                            stride=4,
                            padding=1,
                            )
        # s is the "a grid of pooling units spaced s pixels apart" which sounds like stride
        # z is the "neighborhood of size z × z centered at the location of the pooling unit"
        # here, s = 2 and z = 3 according to Alex Krizhevsky
        self.maxpool = MaxPool2d(kernel_size = 3, stride = 2)
        self.lrn = nn.LocalResponseNorm(size = 5, alpha=1e-4, beta=0.75, k=2)
        self.layer_2 = Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5)
        self.layer_3 = Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3)
        self.layer_4 = Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3)
        self.layer_5 = Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3)
        self.fc_1 = Linear(in_features=1024, out_features=4096)
        self.fc_2 = Linear(in_features=4096, out_features=4096)
        self.fc_3 = Linear(in_features=4096, out_features=num_classes)
        self.relu = ReLU()
        self.softmax = Softmax(dim = 1)
        self.sigmoid = Sigmoid()
        self.dropout = Dropout(p=0.5)

    def forward(self, input: torch.Tensor) -> torch.Tensor: 

        # feature extraction for layers 1 -5
        output = self.layer_1(input)
        output = self.relu(output)
        output = self.lrn(output)
        output = self.maxpool(output)

        output = self.layer_2(output)
        output = self.relu(output)
        output = self.lrn(output)
        output = self.maxpool(output)

        output = self.layer_3(output)
        output = self.relu(output)
        output = self.layer_4(output)
        output = self.relu(output)

        output = self.layer_5(output)
        output = self.relu(output)
        output = self.lrn(output)
        output = self.maxpool(output)

        # FCN part
        output = torch.flatten(output, 1)
        # print (output.shape)
        output = self.fc_1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc_2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc_3(output)
        output = self.relu(output)

        # output = self.softmax(output)
        # apply sigmoid because want to check for confidence
        output = self.sigmoid(output)

        return output 

if __name__ == "__main__":
    import numpy as np
    import torch
    # from torchsummary import summary

    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.rand(5, 3, 227, 227).astype("float32")
    X = torch.tensor(X)

    model = AlexNet()

    # summary(model, (3, 227, 227))
    print(model.forward(X).shape)
    print(model.forward(X))
