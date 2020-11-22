import torch
from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, in_size:int, hidden_size:int, out_size:int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x

    def forward(self, x): return x + self.convblock(x) # skip connection

class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.res1 = ResBlock(3, 8, 3)
        self.res2 = ResBlock(3, 32, 3)
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.fc1 = nn.Linear(32 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        #3x32x32
        x = self.res1(x)
        #3x32x32
        x = self.res2(x)
        #3x32x32
        x = F.max_pool2d(F.relu(x), 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #32x12x12
        x = x.view(-1, 32*12*12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x,dim=-1)

class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(3, 6, 5, padding=2)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(6, 16, 5)

        # Fully connected layer
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        x = F.relu(self.conv1(x))
        # max-pooling with 2x2 grid
        x = F.max_pool2d(x, 2)
        # convolve, then perform ReLU non-linearity
        x = F.relu(self.conv2(x))
        # max-pooling with 2x2 grid
        x = F.max_pool2d(x, 2)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        x = x.view(-1, 16*5*5)
        # FC-1, then perform ReLU non-linearity
        x = F.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = F.relu(self.fc2(x))
        # FC-3
        x = self.fc3(x)

        return F.log_softmax(x,dim=-1)

class CIFARNet(nn.Module):
   def __init__(self):
        super(CIFARNet, self).__init__()
        self.conv11 = nn.Conv2d(3, 64, 3)
        self.conv12 = nn.Conv2d(64, 64, 3)
        self.conv21 = nn.Conv2d(64, 128, 3)
        self.conv22 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 10)
   def forward(self, x):
       #1, 32, 32
       x = F.relu(self.conv11(x))
       #64, 30, 30
       x = F.relu(self.conv12(x))
       #64, 28, 28
       x = F.max_pool2d(x, (2,2))
       #64, 14, 14
       x = F.relu(self.conv21(x))
       #128, 12, 12
       x = F.relu(self.conv22(x))
       #128, 10, 0
       x = F.max_pool2d(x, (2,2))
       #128, 5, 5
       x = x.view(-1, 128 * 5 * 5)
       x = F.relu(self.fc1(x))
       x = self.fc2(x)
       return F.log_softmax(x, dim=-1)

class VWNet(nn.Module):
    def __init__(self, outputSize):
         super(VWNet, self).__init__()
         self.conv11 = nn.Conv2d(3, 64, 3)
         self.conv12 = nn.Conv2d(64, 64, 3)
         self.conv21 = nn.Conv2d(64, 128, 3)
         self.conv22 = nn.Conv2d(128, 128, 3)
         self.fc1 = nn.Linear(128 * 5 * 5, 256)
         self.fc2 = nn.Linear(256, outputSize)
    def forward(self, x):
        #1, 32, 32
        x = F.relu(self.conv11(x))
        #64, 30, 30
        x = F.relu(self.conv12(x))
        #64, 28, 28
        x = F.max_pool2d(x, (2,2))
        #64, 14, 14
        x = F.relu(self.conv21(x))
        #128, 12, 12
        x = F.relu(self.conv22(x))
        #128, 10, 0
        x = F.max_pool2d(x, (2,2))
        #128, 5, 5
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


#A Model to simulate ensembling of voting
class ensembleModel(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(ensembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        # self.modelD = modelD
        self.fc1 = nn.Linear(30, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        # x4 = self.modelD(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class ensembleModel4(nn.Module):
    def __init__(self, modelA, modelB, modelC, modelD):
        super(ensembleModel4, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.fc1 = nn.Linear(40, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x4 = self.modelD(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

def freeze_models(model):
    for name, p in model.named_parameters():
        if ('modelA' in name) or ('modelB' in name) or ('modelC' in name) or ('modelD' in name) :
            p.requires_grad = False

def unfreeze_models(model):
    for name, p in model.named_parameters():
        if ('modelA' in name) or ('modelB' in name) or ('modelC' in name) or ('modelD' in name) :
            p.requires_grad = True
