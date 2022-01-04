import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Please read the free response questions before starting to code.

def run_model(model,running_mode='train', train_set=None, valid_set=None, test_set=None,
    batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):
    """
    This function either trains or evaluates a model.

    training mode: the model is trained and evaluated on a validation set, if provided.
                   If no validation set is provided, the training is performed for a fixed
                   number of epochs.
                   Otherwise, the model should be evaluted on the validation set
                   at the end of each epoch and the training should be stopped based on one
                   of these two conditions (whichever happens first):
                   1. The validation loss stops improving.
                   2. The maximum number of epochs is reached.

    testing mode: the trained model is evaluated on the testing set

    Inputs:

    model: the neural network to be trained or evaluated
    running_mode: string, 'train' or 'test'
    train_set: the training dataset object generated using the class MyDataset
    valid_set: the validation dataset object generated using the class MyDataset
    test_set: the testing dataset object generated using the class MyDataset
    batch_size: number of training samples fed to the model at each training step
    learning_rate: determines the step size in moving towards a local minimum
    n_epochs: maximum number of epoch for training the model
    stop_thr: if the validation loss from one epoch to the next is less than this
              value, stop training
    shuffle: determines if the shuffle property of the DataLoader is on/off

    Outputs when running_mode == 'train':

    model: the trained model
    loss: dictionary with keys 'train' and 'valid'
          The value of each key is a list of loss values. Each loss value is the average
          of training/validation loss over one epoch.
          If the validation set is not provided just return an empty list.
    acc: dictionary with keys 'train' and 'valid'
         The value of each key is a list of accuracies (percentage of correctly classified
         samples in the dataset). Each accuracy value is the average of training/validation
         accuracies over one epoch.
         If the validation set is not provided just return an empty list.

    Outputs when running_mode == 'test':

    loss: the average loss value over the testing set.
    accuracy: percentage of correctly classified samples in the testing set.

    Summary of the operations this function should perform:
    1. Use the DataLoader class to generate trainin, validation, or test data loaders
    2. In the training mode:
       - define an optimizer (we use SGD in this homework)
       - call the train function (see below) for a number of epochs untill a stopping
         criterion is met
       - call the test function (see below) with the validation data loader at each epoch
         if the validation set is provided

    3. In the testing mode:
       - call the test function (see below) with the test data loader and return the results

    """

    optimizer=optim.SGD(model.parameters(),lr=learning_rate)
    
    if train_set != None:
        train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    if valid_set != None:
        valid_loader=DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle)
    if test_set != None :
        test_loader =DataLoader(test_set,  batch_size=batch_size, shuffle=shuffle)
    
    if running_mode == 'train':
        loss_train = []
        acc_train  = []
        loss_valid = []
        acc_valid  = []
        
        count=0
        valid_stop=1000
        for epoch in range(n_epochs) : # loop over the dataset multiple times
            running_loss = 0.0
            model, _est_loss_train, _est_acc_train = _train(model,train_loader,optimizer)
            loss_train.append(_est_loss_train)
            acc_train.append(_est_acc_train)

            if valid_set != None:
                _est_loss_valid, _est_acc_valid = _test(model,valid_loader)
                loss_valid.append(_est_loss_valid)
                acc_valid.append(_est_acc_valid)
                
                if (valid_stop-_est_loss_valid) < stop_thr:
                    break
                else:
                    valid_stop=_est_loss_valid
            count+=1

        print('the eposhc are ',count)
                    
        loss_train_ret={'train':loss_train, 'valid':loss_valid}
        acc_train_ret={'train':acc_train, 'valid':acc_valid}
        return model, loss_train_ret, acc_train_ret
    elif running_mode == 'test' :
        _est_loss_test, _est_acc_test = _test(model,test_loader)
        loss_train_ret=_est_loss_test
        loss_train_acc=_est_acc_test
        return loss_train_ret,  loss_train_acc
  

def _train(model,data_loader,optimizer,device=torch.device('cpu')):

    """
    This function implements ONE EPOCH of training a neural network on a given dataset.
    Example: training the Digit_Classifier on the MNIST dataset
    Use nn.CrossEntropyLoss() for the loss function


    Inputs:
    model: the neural network to be trained
    data_loader: for loading the netowrk input and targets from the training dataset
    optimizer: the optimiztion method, e.g., SGD
    device: we run everything on CPU in this homework

    Outputs:
    model: the trained model
    train_loss: average loss value on the entire training dataset
    train_accuracy: average accuracy on the entire training dataset
    """

    train_loss = 0.0
    correct=0.0
    total=0.0
    lossCE=nn.CrossEntropyLoss()
    for i, data in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
#         print('I am here')
#         print(i, inputs,labels)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.float())
#         print('TYPE output',type(outputs), 'labels',type(labels),'\n')
#         print(outputs,labels)
        loss = lossCE(outputs, labels.long())
        loss.backward()
        optimizer.step()
        
    
        # count statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    train_loss=train_loss/(i+1) 
    acc=100*correct/total
    return model, train_loss, acc


def _test(model, data_loader, device=torch.device('cpu')):
    """
    This function evaluates a trained neural network on a validation set
    or a testing set.
    Use nn.CrossEntropyLoss() for the loss function

    Inputs:
    model: trained neural network
    data_loader: for loading the netowrk input and targets from the validation or testing dataset
    device: we run everything on CPU in this homework

    Output:
    test_loss: average loss value on the entire validation or testing dataset
    test_accuracy: percentage of correctly classified samples in the validation or testing dataset
    """

    test_loss = 0.0
    correct=0.0
    total=0.0
    lossCE=nn.CrossEntropyLoss()
    for i, data in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
#         print('I am here')
#         print(i, inputs,labels)

        # forward + backward + optimize
        outputs = model(inputs.float())
        loss = lossCE(outputs, labels.long())
        
    
        # count statistics
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    test_loss=test_loss/(i+1) 
    acc=100*correct/total
    return test_loss, acc

