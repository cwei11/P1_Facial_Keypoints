## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # According to what is listed in notbook2, the input size will be gray and 224x224 size
        # Then the output will be 224 - 5 + 1==220. (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        # After relu, then use 2x2 to minimize output size
        self.pool1 = nn.MaxPool2d(2, 2)
        #After above step, then the output will become (32, 110, 110)
        
        #then after this step, I thinking a maxpool of 2x2 should be calculate, now the input is 
        
        self.conv2 = nn.Conv2d(32, 64, 3) 
        # Then the size will be 110 - 2 = 106, (64,108,108)
        
        self.pool2 = nn.MaxPool2d(2,2)
        #output will become (64, 54, 54)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        # Then the output size will be 54 - 2 = 52, (128, 52, 52)
        
        self.pool3 = nn.MaxPool2d(2,2)
        # Then the output will be come (128, 26, 26)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        # Then the output size will become 24, (256, 24, 24)
        
        self.pool4 = nn.MaxPool2d(2,2)
        # Output will become (256, 12, 12)
        
        self.conv5 = nn.Conv2d(256, 512, 1)
        # Add extra layer to introduce more non-linear property according to materia 
        #from internet
        # Then the output size will be 12 - 1 +1 = 12, (512, 12, 12)
        
        self.pool5 = nn.MaxPool2d(2,2)
        # Then the output will be come (128, 6, 6)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # Using batch norm to make sure future data processing have enough accuracy 
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.batchNorm4 = nn.BatchNorm2d(256)
        self.batchNorm5 = nn.BatchNorm2d(512)
        
        
        # Added batch norm for full connected layer
        self.batchNormfc1 = nn.BatchNorm1d(1000)
        self.batchNormfc2 = nn.BatchNorm1d(500)
        
        # This is the sugguestion from Udacity advisor 
        self.fullCL1 = torch.nn.Linear(512 * 6 * 6, 1000)
        self.fullCL2 = torch.nn.Linear(1000, 500)
        self.fullCL3 = torch.nn.Linear(500, 136)
        
        # Set up drop out layer to 0.2 possibility 
        self.drop = nn.Dropout(p=0.2)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool1(F.relu(self.batchNorm1(self.conv1(x))))
        x = self.drop(x)
        x = self.pool2(F.relu(self.batchNorm2(self.conv2(x))))
        x = self.drop(x)
        x = self.pool3(F.relu(self.batchNorm3(self.conv3(x))))
        x = self.drop(x)
        x = self.pool4(F.relu(self.batchNorm4(self.conv4(x))))
        x = self.drop(x)
        x = self.pool5(F.relu(self.batchNorm5(self.conv5(x))))
        x = self.drop(x)
        
        x = x.view(x.size(0), -1)
        
    
        x = F.relu(self.batchNormfc1(self.fullCL1(x)))
        x = self.drop(x)
        
        x =self.drop(F.relu(self.batchNormfc2(self.fullCL2(x))))
        
        x = self.fullCL3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
