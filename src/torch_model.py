import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.data_processing import word_to_tensor
from sklearn.model_selection import train_test_split
from src.dataset_management import create_training_set
import torchvision
import torchvision.transforms as transforms
import pandas as pd
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


# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# joined = create_training_set()
joined = pd.read_csv(f"{base_data_path}../training_set.csv")
# joined.to_csv

feats = np.array([i for i in joined.img])
labels = np.array(joined.label)

X_train, X_test, y_train, y_test = train_test_split(
    feats, labels, test_size=0.33, random_state=42, stratify=labels)


print("split data")
print(feats[0])
print(labels[0])


net = Net()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

def train():
    for epoch in range(2):
        total_loss = 0


        for i, data in enumerate(zip(feats, labels)):

            feat, label = data

            optimizer.zero_grad()

            output = net.forward(feat)
            loss = nn.CrossEntropyLoss(output, label)
            loss.backward()
            optimizer.step()

            print(loss)


train()