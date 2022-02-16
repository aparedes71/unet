
import torch.nn as nn
import torch.nn.functional as F

class unet (nn.Module):
    def __init__(self,input_features=1):
        super().__init__()
        #ENCODER
        self.conv1 = nn.Conv2d(input_features,64,3,1) #1 feature in indicates grayscale
        self.conv2 = nn.Conv2d(64,64,3,1)
        #max pool this output
        self.conv3 = nn.Conv2d(128,128,3,1)
        self.conv4 = nn.Conv2d(128,128,3,1)
        #max pool this output
        self.conv5 = nn.Conv2d(256,256,3,1)
        self.conv6 = nn.Conv2d(256,256,3,1) 
        #max pool this output
        self.conv7 = nn.Conv2d(512,512,3,1)
        self.conv8 = nn.Conv2d(512,512,3,1)
        #max pool this output
        self.conv9 = nn.Conv2d(1024,1024,3,1)
        self.conv10 = nn.Conv2d(1024,1024,3,1)

        #DECODER 
        self.upconv1 = nn.ConvTranspose2d(1024,1024,2,1)
        self.conv11 = nn.Conv2d(1024,512,3,1)
        self.conv12 = nn.Conv2d(512,512,3,1)

        self.upconv2 = nn.ConvTranspose2d(512,512,2,1)
        self.conv13 = nn.Conv2d(512,256,3,1)
        self.conv14 = nn.Conv2d(256,256,3,1)

        self.upconv3 = nn.ConvTranspose2d(256,256,2,1)
        self.conv15 = nn.Conv2d(256,128,3,1)
        self.conv16 = nn.Conv2d(128,128,3,1)

        self.upconv4 = nn.ConvTranspose2d(128,128,2,1)
        self.conv17 = nn.Conv2d(128,64,3,1)
        self.conv18 = nn.Conv2d(64,64,3,1)

        self.conv19 = nn.Conv2d(64,2,3,1)
        self.conv20 = nn.Conv2d(2,1,1,1)
            
    def forward(self,X):
        # ENCODER
        X = F.relu (self.conv1(X))
        X1 = X = F.relu (self.conv2(X))
        X = F.max_pool2d(X,2,2)

        X = F.relu (self.conv3(X))
        X2 = X = F.relu (self.conv4(X))
        X = F.max_pool2d(X,2,2)

        X = F.relu (self.conv5(X))
        X3 = X = F.relu (self.conv6(X))
        X = F.max_pool2d(X,2,2)

        X = F.relu (self.conv7(X))
        X4 = X = F.relu (self.conv8(X))
        X = F.max_pool2d(X,2,2)

        X = F.relu (self.conv9(X))
        X = F.relu (self.conv10(X))
        X = F.max_pool2d(X,2,2)

        #DECODER
        #need to figure proper way of implementing the copy and crop functionality