import torch
import torch.nn as nn
import torch.nn.functional as F



class Base_block(nn.Module):
    def __init__(self, in_channels, out_channels, k: int, R: int, stride):
    # k是卷积核大小，R是方法倍数
        super().__init__()
        self.block = nn.ModuleList([nn.Conv3d(in_channels, in_channels, kernel_size=(k, k, k), stride=stride, padding=k // 2),
                                    nn.GroupNorm(in_channels, in_channels),
                                    nn.Conv3d(in_channels, in_channels*R, kernel_size=(1, 1, 1), stride=1, padding=0),
                                    nn.GELU(),
                                    nn.Conv3d(in_channels*R, out_channels, kernel_size=(1, 1, 1), stride=1, padding=0)
                                    ])

    def forward(self, x):
        residual = x
        for laryer in self.block:
            x = laryer(x)
        x = x + residual
        return x


class MedNeXt(Base_block):
    def __init__(self, in_channels, k, R):
        super().__init__(in_channels, in_channels, k, R, 1)


class Down(Base_block):
    def __init__(self, in_channels, k, R):
        super().__init__(in_channels, 2 * in_channels, k, R, 2)
        self.add = nn.Conv3d(in_channels, 2 * in_channels, kernel_size=(1, 1, 1), stride=2, padding=0)
        self.mednext = MedNeXt(2 * in_channels, k, R)

    def forward(self, x):
        residual = self.add(x)
        for laryer in self.block:
            x = laryer(x)
        x = x + residual
        x = self.mednext(x)
        return x


class UP(Base_block):
    def __init__(self, in_channels, k, R):
        super().__init__(in_channels, in_channels//2, k, R, 2)
        self.block[0] = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(k, k, k), stride=2, padding=k // 2)
        self.add = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.mednext = MedNeXt(in_channels//2, k, R)

    def forward(self, x1, x2):
        residual = self.add(x1)
        for laryer in self.block:
            x1 = laryer(x1)
        x1 = x1 + residual
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = x1 + x2
        x = self.mednext(x)
        return x


def DS(channels, n_classes):
    nn.Sequential(
        nn.Conv3d(channels, n_classes, 1, 1),
        nn.GroupNorm(n_classes, n_classes),
        nn.GELU(),
        nn.ConvTranspose3d(n_classes, n_classes, 3, padding=1)
    )

if __name__ == "__main__":
    print("hahaha")
    # m = UP(32, 3, 2)
    # x = torch.randn(10, 32, 10, 50, 100)
    # output = m(x)
    # # m = nn.Conv3d(16, 33, 3, stride=1, padding=1)
    # # input = torch.randn(20, 16, 10, 50, 100)
    # # output = m(input)
    # print(output.shape)


