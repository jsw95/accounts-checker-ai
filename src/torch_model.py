import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io

from src.data_processing import generate_char_dict
from src.data_processing import resize_img, binary_threshold

base_data_path = "/home/jack/Workspace/data/accounts/images/"

device = torch.cuda.device("cuda:0" if torch.cuda.is_available() else "cpu")


# net.to(device)
# inputs, labels = inputs.to(device), labels.to(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 70)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 13456)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

imgs, labs = [], []
for file in os.listdir("/home/jack/Workspace/data/accounts/English/Img/allImgs"):
    if file.endswith('.png'):
        lab = re.match(r'img([0-9]+)', file).group(1)
        img = io.imread(f"/home/jack/Workspace/data/accounts/English/Img/allImgs/{file}", as_grey=True)
        img = resize_img(img, (128, 128))
        img = binary_threshold(img)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(torch.float32)
        imgs.append(img)
        labs.append(lab)

char_dict = generate_char_dict()


def train():
    print("Training")
    for epoch in range(2):
        total_loss = 0

        # for i, data in enumerate(trainloader, 0):
        for feat, label in zip(imgs, labs):
            label = torch.Tensor([int(label)]).to(torch.int64)
            optimizer.zero_grad()

            output = net(feat)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            print(loss)


train()
