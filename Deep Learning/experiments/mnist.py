import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
from data.load_data import load_mnist_data
from data.my_dataset import MyDataset
from src.run_model import run_model,_train,_test
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.h1 = nn.Linear(28*28, 128)
        self.h2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.out(x)
        return x



example = [500, 1000, 1500, 2000]
time_list = []
loss_list = []
acc_list=[]
for n in example:

    train_features, _, train_targets, _ = load_mnist_data(
        10, fraction=1.0, examples_per_class=int(n / 10))

    _, test_features, _, test_targets = load_mnist_data(
        10, fraction=0.0, examples_per_class=int(1000 / 10))


    # print(train_features.shape[0],train_features.shape[1])
    # print(test_features.shape[0],test_features.shape[1])
    # train_targets=list(train_targets)
    # print(train_targets.count(0))
    # print(train_targets.count(2))
    # print(train_targets.count(4))
    # print(train_targets.count(8))

    train_set = MyDataset(train_features, train_targets)
    # train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    max_epochs = 100

    test_set = MyDataset(test_features, test_targets)
    # test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True)

    model=Net()
    start_time = time.time()
    _, _est_loss, _est_acc = run_model(model,running_mode='train', train_set=train_set, batch_size=10, learning_rate=0.01,
        n_epochs=100, shuffle=True)
    
    time_list.append(time.time() - start_time)

    loss, acc = run_model(model,running_mode='test', train_set=None, valid_set=None, test_set=test_set,
    batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True)
    print(loss)
    
    loss_list.append(loss)
    acc_list.append(acc)

plt.figure()
plt.plot(example, time_list)
plt.title("Training Time vs. Num. Training Examples")
plt.xlabel('Num. Training Examples')
plt.ylabel("Training Time for 100 epochs (s)")
plt.grid(True)
plt.savefig('problem_1a.png')


plt.figure()
plt.plot(example, acc_list)
plt.title("Testing Accuracy vs. Num. Training Examples")
plt.xlabel('Num. Training Examples')
plt.ylabel("Testing Accuracy  (%)")
plt.grid(True)
# plt.show()
plt.savefig('problem_1c.png')


