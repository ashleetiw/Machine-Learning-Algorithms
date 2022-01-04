
# loading code from the assignment
from src.models import Large_Dog_Classifier
from data.my_dataset import MyDataset
from src.run_model import run_model
from data.dogs import DogsDataset
from data.my_dataset import MyDataset

# torch code that we need
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import os

# students will specify the file path here
model_weights_path = 'experiments/large-CNN'
model=Large_Dog_Classifier()

model.load_state_dict(torch.load(model_weights_path))

data=DogsDataset('./data/DogSet', num_classes=10)

print(model.eval())

trainx,trainy=data.get_train_examples()
# print(trainx.shape[0],trainx.shape[1])

testx,testy=data.get_test_examples()
# print(testx.shape[0],testx.shape[1])

validx,validy=data.get_validation_examples()
# print(validx.shape[0],validx.shape[1])

# grab an image from the testing set, it doesn't have to be random.
# if you're having trouble finding a good image we found 86,
# img_num = np.random.randint()
filter_image = testx[86]

# we need to put the image into a tensor since the network expects input to come in batches
# our batch size will be 1
filter_image = torch.tensor([filter_image])

# function to plot our image 
def imshow(img):
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.axis('off')
    plt.show()
filter_image = filter_image/2 +.5

# display the image, students will write this
# imshow(filter_image[0])
# print(filter_image.shape)


# # permute the image just as you did in the homework
filter_image = filter_image.reshape(1,3,64,64)


# # # pass the image through the network and save the output to a variable
filter1 = model.conv1(filter_image)
filter2 = model.conv2(F.relu(filter1))
filter3 = model.conv3(F.relu(filter2))
filter4 = model.conv4(F.relu(filter3))
filter5 = model.conv5(F.relu(filter4))
filter6 = model.conv6(F.relu(filter5))
filter7 = model.conv7(F.relu(filter6))
# # #we need to detach the gradient variable and convert the tensors to numpy arrays
filter1 = filter1.detach().numpy()
filter2 = filter2.detach().numpy()
filter3 = filter3.detach().numpy()
filter4 = filter4.detach().numpy()
filter5 = filter5.detach().numpy()
filter6 = filter6.detach().numpy()
filter7 = filter7.detach().numpy()

# # looking at the image shapes here may be helpful in one of the follow up questions
print(f'filter1: {filter1.shape}')
print(f'filter2: {filter2.shape}')
print(f'filter3: {filter3.shape}')
print(f'filter4: {filter4.shape}')
print(f'filter5: {filter5.shape}')
print(f'filter6: {filter6.shape}')
print(f'filter7: {filter7.shape}')


# # filter1: (1, 4, 64, 64)
# # filter2: (1, 6, 64, 64)

def graph_filters(filters):
    """
        graph_filter - graphs a list of images which have been taken out at various stages of a CNN
        
        args:
            filters (list) - list containg the output of a convolutional layer in the network
        
    """
    for filter_mat in filters:
        channels = filter_mat.shape[1]
        display_grid = np.zeros((filter_mat.shape[3], channels * filter_mat.shape[3]))
        # print(display_grid.shape)
        for i in range(channels):
            x = filter_mat[0, i, :, :]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            display_grid[:, i * filter_mat.shape[3] : (i + 1) * filter_mat.shape[3]] = x
        
        scale = 40. / channels
        plt.grid(False)
        plt.figure(figsize=(scale * channels, scale))
        plt.axis('off')
        plt.imshow(display_grid*255, aspect='auto', cmap='viridis')
    
    plt.show()

graph_filters([filter1,filter2,filter3,filter4,filter5,filter6,filter7])


