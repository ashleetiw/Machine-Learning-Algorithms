import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Please read the free response questions before starting to code.
#
# Note: Avoid using nn.Sequential here, as it prevents the test code from
# correctly checking your model architecture and will cause your code to
# fail the tests.

class Digit_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """
    def __init__(self):
        super(Digit_Classifier, self).__init__()
        D_in, H1, H2, D_out = 28*28, 128, 64, 10
        self.layer1=torch.nn.Linear(D_in, H1)
        self.layer2=torch.nn.Linear(H1,H2)
        self.layerOut=torch.nn.Linear(H2, D_out)

    def forward(self, inputs):
        x = F.relu(self.layer1(inputs))
        x = F.relu(self.layer2(x))
        x = self.layerOut(x)
        return x


class Dog_Classifier_FC(nn.Module):
    """
    This is the class that creates a fully connected neural network for classifying dog breeds
    from the DogSet dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)
    Activation function: ReLU for both hidden layers

    """

    def __init__(self):
        super(Dog_Classifier_FC, self).__init__()
        D_in, H1, H2, D_out = 12288, 128, 64, 10
        self.layer1=nn.Linear(D_in, H1)
        self.layer2=nn.Linear(H1,H2)
        self.layerOut=nn.Linear(H2, D_out)

    def forward(self, inputs):
        inn = inputs.reshape(-1, 12288)
        x = F.relu(self.layer1(inn))
        x = F.relu(self.layer2(x))
        x = self.layerOut(x)
        return x

# raise NotImplementedError()


class Dog_Classifier_Conv(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying dog breeds
    from the DogSet dataset.
    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 10)
    Activation function: ReLU for both hidden layers
    There should be a maxpool after each convolution.
    The sequence of operations looks like this:
        1. Apply convolutional layer with stride and kernel size specified
            - note: uses hard-coded in_channels and out_channels
            - read the problems to figure out what these should be!
        2. Apply the activation function (ReLU)
        3. Apply 2D max pooling with a kernel size of 2
    Inputs:
    kernel_size: list of length 2 containing kernel sizes for the two convolutional layers
                 e.g., kernel_size = [(3,3), (3,3)]
    stride: list of length 2 containing strides for the two convolutional layers
            e.g., stride = [(1,1), (1,1)]
    """

    def __init__(self, kernel_size, stride):
        super(Dog_Classifier_Conv, self).__init__()
           # 1 input image channel, 6 output channels, 3x3 square convolution
        self.layer1 = nn.Conv2d(3,16, kernel_size[0], stride=stride[0])
        self.layer2 = nn.Conv2d(16,32, kernel_size[1], stride=stride[1])
        self.layerOut=nn.Linear(5408, 10)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, inputs):
        inn=inputs.permute((0,3,1,2))
        x = F.relu(self.layer1(inn))
        x=self.max_pool(x)
        x = F.relu(self.layer2(x))
        x=self.max_pool(x)
        x=torch.flatten(x,1)
        x = self.layerOut(x)
        return x

class Large_Dog_Classifier(nn.Module):
    """
    This is the class that creates a convolutional neural network for visualizing an image as it is passed through a convolutional neural network.

    """

    def __init__(self):
        super(Large_Dog_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=(3,3), padding=1)
        self.conv3 = nn.Conv2d(6, 8, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(8, 10, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(10, 12, kernel_size=(3, 3))
        self.conv6 = nn.Conv2d(12, 14, kernel_size=(3, 3))
        self.conv7 = nn.Conv2d(14, 16, kernel_size=(3, 3), stride=(2,2))
        self.fc1 = nn.Linear(11664, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, input):
        input = input.permute((0, 3, 1, 2))
        input = F.relu(self.conv1(input))
        input = F.relu(self.conv2(input))
        input = F.relu(self.conv3(input))
        input = F.relu(self.conv4(input))
        input = F.relu(self.conv5(input))
        input = F.relu(self.conv6(input))
        input = F.relu(self.conv7(input))
        input = F.relu(self.fc1(input.view(-1, 11664)))
        input = self.fc2(input)

        return input


       


