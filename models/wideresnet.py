# import torch
from torch import nn
from torchsummary import summary


class Pipe(nn.Module):

    def __init__(self, input_dims, filters, downsample=False):
        super(Pipe, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_dims)
        self.bn2 = nn.BatchNorm2d(filters)
        self.cn1 = nn.Conv2d(input_dims, filters, kernel_size=3, padding=1, stride=2 if downsample else 1)
        # we want to downsample only in the first Pipe
        self.cn2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.af = nn.ReLU(inplace=True) # helps with memory but can cause problems

    def forward(self, x):
        _ = self.bn1(x)
        _ = self.af(_)
        _ = self.cn1(_)
        _ = self.bn2(_)
        _ = self.af(_)
        return self.cn2(_)


class Residual(nn.Module):

    def __init__(self, layers, input_dims, filters, downsample = True):
        super(Residual, self).__init__()
        self.filters = filters
        self.layers = nn.ModuleList([Pipe(input_dims, filters, downsample=downsample)])
        for i in range(1, layers):
            self.layers.append(Pipe(filters, filters))
        self.same_dims = True
        self.scale = nn.Identity()  # check if we need arguments
        if input_dims != filters:
            self.scale = nn.Conv2d(input_dims, filters, kernel_size=1, stride=2 if downsample else 1)

    def forward(self, x):
        _ = self.layers[0](x)
        _ = _ + self.scale(x)  # scaling only the first
        for i in range(1, len(self.layers)):
            _ = self.layers[i](_) + _
        return _


class WideResNet(nn.Module):

    def __init__(self, input_dims, n, k, classes, first_layer_dims=16):
        super(WideResNet, self).__init__()
        self.k = k
        self.N = (n-4)//6  # 1 initial layer and 3 for scalling the networks input
        # 6 is 3*2convolutions
        self.dimensions_of_next_convs = [16*k, 32*k, 64*k]
        self.layers = nn.ModuleList([nn.Conv2d(input_dims, first_layer_dims, kernel_size=3, padding=1)])
        cur_dims = 16
        downsample = False
        for ftr in self.dimensions_of_next_convs:
            self.layers.append(Residual(self.N, cur_dims, ftr, downsample=downsample))
            downsample = True
            cur_dims = ftr
        self.layers.append(nn.AvgPool2d(kernel_size=8))

        self.fc = nn.Linear(self.dimensions_of_next_convs[-1], classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        _ = self.layers[0](x)
        for i in range(1, len(self.layers)):
            _ = self.layers[i](_)
        _ = _.view(-1, self.dimensions_of_next_convs[-1])
        return self.fc(_)


if __name__ == '__main__':
    wrn = WideResNet(3, 28, 2, 10)
    print("Just testing the model")
    summary(wrn, (3, 32, 32))
