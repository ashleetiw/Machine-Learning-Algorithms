from data.dogs import DogsDataset
import matplotlib.pyplot as plt
import numpy as np
from data.my_dataset import MyDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.run_model import run_model,_train,_test
from src.models import Dog_Classifier_Conv
import time



data=DogsDataset('./data/DogSet', num_classes=10)

trainx,trainy=data.get_train_examples()
# print(trainx.shape[0],trainx.shape[1])

testx,testy=data.get_test_examples()
# print(testx.shape[0],testx.shape[1])

validx,validy=data.get_validation_examples()
# print(validx.shape[0],validx.shape[1])


time_list = []
train_loss_list=[]
valid_loss_list=[]
train_accuracy_list=[]
valid_accuracy_list=[]

train_set = MyDataset(trainx, trainy)
test_set = MyDataset(testx,testy)
valid_set = MyDataset(validx,validy)
model = Dog_Classifier_Conv([(5,5),(5,5)],[(1,1),(1,1)])
start_time = time.time()
model, train_loss, train_accuracy = run_model(model, running_mode='train', train_set=train_set,
                                                  valid_set=valid_set,
                                                  test_set=test_set,
                                                  batch_size=10, learning_rate= 1e-5, n_epochs=100,
                                                  stop_thr=1e-4,
                                                  shuffle=True)
c=0
for param in model.parameters():
        c+=int(param.numel())
# print(model.parameters())
print('final model parmaters  ',c)
end_time = time.time()
running_time = end_time - start_time
time_list.append(running_time)

train_loss_list.append(np.mean(train_loss['train']))
valid_loss_list.append(np.mean(train_loss['valid']))
train_accuracy_list.append(np.mean(train_accuracy['train']))
valid_accuracy_list.append(np.mean(train_accuracy['valid']))

test_loss, test_accuracy = run_model(model, running_mode='test', train_set=train_set,valid_set=valid_set,
                                                  test_set=test_set,
                                        batch_size=10, learning_rate=1e-5, n_epochs=100, stop_thr=1e-4,
                                        shuffle=True)

plt.figure()
plt.plot(range(len(train_loss['train'])), train_loss['train'], label='Training Loss')
plt.plot(range(len(train_loss['valid'])), train_loss['valid'], label='Validation Loss')
plt.title('Training and Validation Loss Vs. Epoch Number')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.savefig("dogconv-loss.png")
plt.show()

plt.figure()
plt.plot(range(len(train_accuracy['train'])), train_accuracy['train'], label='Training Accuracy')
plt.plot(range(len(train_accuracy['valid'])), train_accuracy['valid'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy Vs. Epoch Number')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.savefig("dogconv-accuracy.png")
plt.show()

print("The accuracy of your model on the testing set:", test_accuracy)

