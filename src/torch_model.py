import os
import torchvision.transforms as transforms

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from src.data_processing import generate_char_dict, word_to_tensor
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

base_data_path = "/home/jack/Workspace/data/accounts/images/"

"""
Steps to perform image classification using Neural Networks
1. Load and normalizing the training and test datasets 
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data
"""

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.9
log_interval = 10

random_seed = 1


device = torch.cuda.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(device)
# net.to(device)
# inputs, labels = inputs.to(device), labels.to(device)

# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2176, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2176)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

imgs = [io.imread(f"/home/jack/Workspace/data/accounts/images/trainset/words/{file}") for file in
        os.listdir("/home/jack/Workspace/data/accounts/images/trainset/words/")[:10] if file.endswith('.jpg')]

imgs = [torch.from_numpy(i).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0) for i in imgs]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

char_dict = generate_char_dict()





def train():
    print("Training")
    for epoch in range(2):
        total_loss = 0

        # for i, data in enumerate(trainloader, 0):
        for i, data in enumerate(imgs, 0):

            # inputs, labels = data
            word = word_tokenize('label')
            print(word)
            labels = torch.Tensor([word]).to(torch.int64)
            inputs = data
            print(inputs.shape)
#
#
#             optimizer.zero_grad()
#
#             output = net(inputs)
#             loss = criterion(output, labels)
#             loss.backward()
#             optimizer.step()
#
#             print(loss)
#
train()