import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io
import pathlib

from src.data_processing import generate_char_dict
from src.data_processing import resize_img, binary_threshold, transform_image_for_training

base_data_path = "/home/jack/Workspace/data/accounts/images/"

print(f"GPU available: {torch.cuda.is_available()}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 120)
        self.fc2 = nn.Linear(120, 76)
        self.fc3 = nn.Linear(76, 58)  # num chars

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 13456)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    print("Training")
    epoch_loss = 0

    for epoch in range(10):
        print(epoch_loss)
        epoch_loss = 0

        # for i, data in enumerate(trainloader, 0):
        for inputs, label in zip(imgs, labs):
            label = torch.Tensor([int(label)]).to(torch.int64)
            inputs, label = inputs.to(device), label.to(device)
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)

    torch.save(net.state_dict(), f'{pathlib.Path(__file__).parent.parent}/models/first_.sav')


if __name__ == '__main__':
    net = Net()
    net.to(device)

    char_dict = generate_char_dict()
    print(char_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    imgs, labs = [], []
    for file in os.listdir("/home/jack/Workspace/data/accounts/English/Img/allImgs")[50:200]:
        if file.endswith('.png'):
            lab = re.match(r'img([0-9]+)', file).group(1)
            img = io.imread(f"/home/jack/Workspace/data/accounts/English/Img/allImgs/{file}", as_grey=True)
            img = transform_image_for_training(img)
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(torch.float32)
            imgs.append(img)

            labs.append(lab)

    train()
