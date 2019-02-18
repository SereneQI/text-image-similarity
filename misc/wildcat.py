import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function, Variable


class WildcatPool2dFunction(Function):
    def __init__(self, kmax, kmin, alpha):
        super(WildcatPool2dFunction, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        self.alpha = alpha

    def get_positive_k(self, k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k > n:
            return int(n)
        else:
            return int(k)

    def forward(self, input):
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        n = h * w  # number of regions

        kmax = self.get_positive_k(self.kmax, n)
        kmin = self.get_positive_k(self.kmin, n)

        sorted, indices = input.new(), input.new().long()
        torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True, out=(sorted, indices))

        self.indices_max = indices.narrow(2, 0, kmax)
        output = sorted.narrow(2, 0, kmax).sum(2).div_(kmax)

        if kmin > 0 and self.alpha is not 0:
            self.indices_min = indices.narrow(2, n - kmin, kmin)
            output.add_(sorted.narrow(2, n - kmin, kmin).sum(2).mul_(self.alpha / kmin)).div_(2)

        self.save_for_backward(input)
        return output.view(batch_size, num_channels)

    def backward(self, grad_output):

        input, = self.saved_tensors

        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        n = h * w  # number of regions

        kmax = self.get_positive_k(self.kmax, n)
        kmin = self.get_positive_k(self.kmin, n)

        grad_output_max = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmax)

        grad_input = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2, self.indices_max,
                                                                                              grad_output_max).div_(
            kmax)

        if kmin > 0 and self.alpha is not 0:
            grad_output_min = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmin)
            grad_input_min = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2,
                                                                                                      self.indices_min,
                                                                                                      grad_output_min).mul_(
                self.alpha / kmin)
            grad_input.add_(grad_input_min).div_(2)

        return grad_input.view(batch_size, num_channels, h, w)


class WildcatPool2d(nn.Module):
    def __init__(self, kmax=1, kmin=None, alpha=1):
        super(WildcatPool2d, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        if self.kmin is None:
            self.kmin = self.kmax
        self.alpha = alpha

    def forward(self, input):
        return WildcatPool2dFunction(self.kmax, self.kmin, self.alpha)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax) + ', kmin=' + str(self.kmin) + ', alpha=' + str(
            self.alpha) + ')'
            
            
            
class ResNet_wildcat(nn.Module):

    def __init__(self, pretrained=True, pretrained_path=None):
        super(ResNet_wildcat, self).__init__()

        resnet = models.resnet152(pretrained=pretrained)

        self.base_layer = nn.Sequential(*list(resnet.children())[:-2])
        self.spaConv = nn.Conv2d(2048, 2400, 1,)

        # add spatial aggregation layer
        self.wldPool = WildcatPool2d(15)
        # Linear layer for imagenet classification
        #self.fc = nn.Linear(2400, 1000)

        # Loading pretrained weights of resnet weldon on imagenet classification
        if pretrained and not pretrained_path is None:
            try:
                self.load_state_dict(torch.load(pretrained_path)['state_dict'])
            except Exception:
                print("Error when loading pretrained resnet weldon")

    def forward(self, x):
        x = self.base_layer(x)
        x = self.spaConv(x)
        x = self.wldPool(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x
            
            
