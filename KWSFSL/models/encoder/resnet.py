# adapted from: https://github.com/roman-vygon/triplet_loss_kws

import torch

from torch import nn
from torch.nn import functional as F


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class CustomSwish(nn.Module):
    def forward(self, input_tensor):
        return Swish.apply(input_tensor)


class Res15(nn.Module):

    def __init__(self, n_maps, dilation = True):
        super().__init__()
        n_maps = n_maps
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.n_layers = n_layers = 13
        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), 
                                    padding=int(2 ** (i // 3)), 
                                    dilation=int(2 ** (i // 3)),
                                    bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                                    bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)

    def forward(self, x):

        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)

        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return x

class Res15_Batch(nn.Module):

    def __init__(self, n_maps, dilation = True):
        super().__init__()
        n_maps = n_maps
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.n_layers = n_layers = 13
        
        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2 ** (i // 3)), dilation=int(2 ** (i // 3)),
                                    bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                                    bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)

    def forward(self, x):
        y = getattr(self, "conv0")(x)
        if hasattr(self, "pool"):
            y = self.pool(y)
        x = y

        for i in range(self.n_layers):

            if i > 0 and  i % 2 == 0:
                x = x + old_x    
            elif i % 2 == 1:
                old_x = y

            y = F.relu(x)

            x = getattr(self, "conv{}".format(i+1))(y)
            x = getattr(self, "bn{}".format(i+1))(x)



        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return x

class Res8(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        n_maps = hidden_size
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.pool = nn.AvgPool2d((3, 4))  # flipped -- better for 80 log-Mels

        self.n_layers = n_layers = 6
        self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module(f'bn{i + 1}', nn.BatchNorm2d(n_maps, affine=False))
            self.add_module(f'conv{i + 1}', conv)

    def forward(self, audio_signal, length=None):

        x = audio_signal.unsqueeze(1)
        x = x.permute(0, 1, 3, 2).contiguous()  # Original res8 uses (time, frequency) format
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, f'conv{i}')(x))
            if i == 0:
                if hasattr(self, 'pool'):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, f'bn{i}')(x)
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        print(x)
        return x.unsqueeze(-2), length
    
class Res15_RELU(nn.Module):

    def __init__(self, n_maps):
        super().__init__()
        n_maps = n_maps
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.n_layers = n_layers = 13
        dilation = True
        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2 ** (i // 3)), dilation=int(2 ** (i // 3)),
                                    bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                                    bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)

    def forward(self, x):

        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)

        x = F.relu(x)
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return x