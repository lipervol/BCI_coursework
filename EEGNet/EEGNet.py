import numpy as np
import torch
from torch import nn
from torch.nn.functional import elu
import torch.optim as op
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt

data_path = "./data"
models_save_path = "./save"


class EEGNet(nn.Module):
    def __init__(self, input_len=500, input_channels=64, output_len=40):
        super(EEGNet, self).__init__()
        self.input_len = input_len
        self.input_channels = input_channels
        self.output_len = output_len

        self.conv1 = nn.Conv2d(1, 16, (input_channels,1), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        self.dropout1 = nn.Dropout(0.25)

        self.padding2 = nn.ZeroPad2d((16,15,1,0))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d((2, 4))
        self.dropout2 = nn.Dropout(0.25)

        self.padding3 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        self.dropout3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(4 * 4 * (input_len // 16), output_len)

    def forward(self, x):
        x = x.view(-1,1,x.size(1),x.size(2))

        x = elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = x.permute(0, 2, 1, 3)
        x = self.dropout1(x)

        x = self.padding2(x)
        x = elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        x = self.padding3(x)
        x = elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.pooling3(x)
        x = self.dropout3(x)

        x = x.view(-1, 4 * 4 * (self.input_len // 16))
        x = self.fc1(x)
        return x


class My_Dataset(Dataset):
    def __init__(self, path, ID, file_num=35, class_num=40):
        super(My_Dataset, self).__init__()
        self.path = path
        self.ID = ID
        self.file_num = file_num
        self.class_num = class_num

        for i in range(self.file_num):
            if (i + 1) == self.ID:
                pass
            else:
                temp = np.load(self.path + "/S%d.npy" % (i + 1))
                try:
                    self.data = np.concatenate((self.data, temp), axis=0)
                except AttributeError:
                    self.data = temp

    def __getitem__(self, idx):
        i, j = idx // self.class_num, idx % self.class_num
        return self.data[i, j], j

    def __len__(self):
        return self.data.shape[0] * self.data.shape[1]


if __name__ == "__main__":

    model_ID = 1

    print("[*]Load data")
    train_dataset = My_Dataset(data_path, model_ID)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    if os.path.exists(models_save_path + "/model%d.pth" % model_ID):
        print("[*]Load model")
        model = torch.load(models_save_path + "/model%d.pth" % model_ID)
    else:
        print("[*]Create model")
        model = EEGNet(500, 64, 40)

    device = torch.device("cuda")
    model = model.to(device)
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = op.Adam(model.parameters(), lr=1e-3)

    train_epochs = 30
    loss_hist = []
    print("[*]Train start")
    for epochs in range(train_epochs):
        model.train()
        for item in train_dataloader:
            data, label = item
            data = data.type(torch.float32)
            data = data.to(device)
            label = label.type(torch.int64)
            label = label.to(device)

            output = model(data)
            loss_out = loss(output, label)

            optimizer.zero_grad()
            loss_out.backward()
            optimizer.step()
        print("[*]Epoch:%d" % (epochs + 1), "\tLoss:", loss_out.item())
        loss_hist.append(loss_out.item())
    print("[*]Train over")

    torch.save(model, models_save_path + "/model%d.pth" % model_ID)
    print("[*]Save model")

    plt.plot([i + 1 for i in range(train_epochs)], loss_hist, 'ro-', alpha=1, linewidth=1, label='Loss')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
