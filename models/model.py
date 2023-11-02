import torch
import torch.nn as nn
import torch.nn.functional as F
# import sys
# sys.path.append("/homes/syli/python/semi_lnm")
from models.unet_parts import *
from models.modules import MedNeXt, Down, UP
from monai.networks.nets import DenseNet121
import os
from collections import OrderedDict
from monai.networks.layers.utils import get_act_layer
from models.resnet import generate_model
from models.vit_pytorch.simple_vit_3d import SimpleViT


class MnistModel(nn.Module): #BaseModel
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

class MultiTaskResNet(nn.Module):
    def __init__(self, input_channels=3):
        super(MultiTaskResNet, self).__init__()
        
        # Load a pre-trained ResNet model
        self.resnet = generate_model(18, n_input_channels=input_channels, conv1_t_size=5)
        
        # Remove the original fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Task 1-specific layers
        self.fc_task1 = nn.Linear(512, 1)
        
        # Task 2-specific layers
        self.fc_task2 = nn.Linear(512, 1)
        self.m = nn.Sigmoid()
        
    def forward(self, x):
        # Forward pass through the ResNet backbone
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        # Task 1 branch
        output_task1 = self.m(self.fc_task1(x))
        
        # Task 2 branch
        output_task2 = self.m(self.fc_task2(x))
        
        return output_task1, output_task2


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class segment(nn.Module):
    def __init__(self, n_channels, n_classes, k, R):
        super(segment, self).__init__()
        init_channels = 32
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.stem = nn.Sequential(nn.Conv3d(n_channels, 32, 1, 1), MedNeXt(32, k, R))
        self.down = []
        for i in range(4):
            self.down.append(Down(init_channels*(2**i), k, R))
        self.down = nn.ModuleList(self.down)
        self.up = []
        for i in range(4, 0, -1):
            self.up.append(UP(init_channels * (2**i), k, R))
        self.up = nn.ModuleList(self.up)
        self.final_super = []
        for i in range(4, -1, -1):
            self.final_super.append(nn.Sequential(
                nn.Conv3d(init_channels * (2**i), init_channels * (2**i), 1, 1),
                nn.GroupNorm(init_channels * (2**i), init_channels * (2**i)),
                nn.GELU(),
                nn.Conv3d(init_channels * (2**i), n_classes, 1, 1)
            ))
        self.final_super = nn.ModuleList(self.final_super)

    def forward(self, x):
        # 正确的写法
        trace = []
        output = []
        size = x.shape[2:]
        x = self.stem(x)
        trace.append(x)
        # 四次下采样
        for layer in self.down:
            x = layer(x)
            # print(x.grad)
            trace.append(x)
        # 四次上采样
        for i, layer in enumerate(self.up):
            x = layer(x, trace[3-i])
            trace.append(x)
        # 四个深监督预测头
        for i, layer in enumerate(self.final_super):
            level_out = F.interpolate(layer(trace[4+i]),
                    size=size, mode='trilinear', align_corners=False)
            output.append(level_out)
        return output
    

class MultiTaskDenseNet(nn.Module):
    def __init__(self, num_classes_task1, num_classes_task2):
        super(MultiTaskDenseNet, self).__init__()

        # Load a pre-trained DenseNet-121 model
        self.densenet = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)

        # Remove the original classification layer
        num_ftrs = self.densenet.class_layers[3].in_features
        self.densenet.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name="relu")),
                    ("pool", nn.AvgPool3d(kernel_size=2, stride=2, padding=1)),
                    ("flatten", nn.Flatten(1)),
                ]
            )
        )

        # Task 1-specific layers
        self.fc_task1 = nn.Linear(num_ftrs, num_classes_task1)

        # Task 2-specific layers
        self.fc_task2 = nn.Linear(num_ftrs, num_classes_task2)

    def forward(self, x):
        # Forward pass through the DenseNet backbone
        features = self.densenet(x)

        # Task 1 branch
        output_task1 = self.fc_task1(features)

        # Task 2 branch
        output_task2 = self.fc_task2(features)

        return output_task1, output_task2
    
class MultiTaskSimpleVIT(nn.Module):
    def __init__(self, input_channels=3):
        super(MultiTaskSimpleVIT, self).__init__()
        
        # Load a pre-trained ResNet model
        self.vit = SimpleViT(
            image_size = 200,
            image_patch_size = 20,
            frames=12, frame_patch_size=3,
            num_classes = 1,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            channels = input_channels
            )
        
        # Remove the original fully connected layer
        self.vit = nn.Sequential(*list(self.vit.children())[:-2])
        # Task 1-specific layers
        self.fc_task1 = nn.Linear(1024, 1)
        
        # Task 2-specific layers
        self.fc_task2 = nn.Linear(1024, 1)
        self.m = nn.Sigmoid()
        
    def forward(self, x):
        # Forward pass through the ResNet backbone
        # x = x.permute(0,1,4,2,3)
        # print(x.shape)
        x = self.vit(x)
        x = x.view(x.size(0), -1)
        # Task 1 branch
        output_task1 = self.m(self.fc_task1(x))
        
        # Task 2 branch
        output_task2 = self.m(self.fc_task2(x))
        
        return output_task1, output_task2


if __name__ == "__main__":
    m = MultiTaskSimpleVIT()
    #med = MedNeXt(2, k, R)
    # down = Down(32, k, R)
    x = torch.randn((4, 3, 12, 200,200), requires_grad=True)
    y = torch.randn((4, 3, 200,200,12), requires_grad=True)
    vit = SimpleViT(
            image_size = 200,
            image_patch_size = 20,
            frames=12, frame_patch_size=3,
            num_classes = 1,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            channels = 3
            )
    # y = torch.randn(2, 2, 100, 100, 10, requires_grad=True)
    # stem = nn.Sequential(nn.Conv3d(2, 32, 1, 1), MedNeXt(32, 3, 2))
    # med = nn.Sequential(stem, down)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x = x.to(device)
    # y = y.to(device)
    # model = m.to(device)

    # output1 = model(x)[3]
    # loss = torch.sum(output1)
    # loss.backward()
    # output2 = model(y)[3]
    # loss = torch.sum(output2)
    # loss.backward()
    #print(vit(x).shape)
    #print(m.vit)
    print(vit)
    print(m.vit)

    # summary(model, (4, 224, 224, 10))

