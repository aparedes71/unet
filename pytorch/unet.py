
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class final_conv_block(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class downsampling_block(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_features, out_features)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class upsampling_block(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_features, in_features // 2, kernel_size=2, stride=2)
        self.conv = conv_block(in_features, out_features)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)     

class unet (nn.Module):
    def __init__(self,input_features=1,num_classes=10):
        super().__init__()
        #ENCODER
        self.input_conv = conv_block(input_features, 64)
        self.down1 = downsampling_block(64,128)
        self.down2 = downsampling_block(128,256)
        self.down3 = downsampling_block(256,512)
        self.down4 = downsampling_block(512,1024)

        #DECODER
        self.up1 = upsampling_block(1024,512)
        self.up2 = upsampling_block(512,256)
        self.up3 = upsampling_block(256,128)
        self.up4 = upsampling_block(128,64)

        self.final = final_conv_block(64,num_classes)
            
    def forward(self,x):
        # ENCODER
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #DECODER
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.final(x)

        return torch.max(x,dim = 0)[0]
