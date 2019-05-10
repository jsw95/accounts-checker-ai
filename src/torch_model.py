import os
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.data_processing import lineToTensor

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net.to(device)
# inputs, labels = inputs.to(device), labels.to(device)

# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(5 * 5 * 20, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



net = Net()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

img = [io.imread(f"/home/jwells/data/accounts/training/{file}") for file in os.listdir("/home/jwells/data/accounts/training/")]


