import numpy as np
import torch
from torch import nn
import torch.optim as op
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt

data_path = "./data"
models_save_path = "./save"


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        self.input_len = 8

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.l_q = nn.Linear(hid_dim, hid_dim)
        self.l_k = nn.Linear(hid_dim, hid_dim)
        self.l_v = nn.Linear(hid_dim, hid_dim)

        self.l_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()

    def forward(self, q, k, v):
        batch_size = q.shape[0]
        x_q = self.l_q(q)
        x_k = self.l_k(k)
        x_v = self.l_v(v)

        x_q = x_q.view(-1, x_q.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        x_k = x_k.view(-1, x_k.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        x_v = x_v.view(-1, x_v.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        x = torch.matmul(x_q, x_k.transpose(-1, -2))
        x = x / self.scale
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(self.dropout(x), x_v)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)

        x = self.l_o(x)
        return x


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, dropout=0.2):
        super(PositionwiseFeedforwardLayer, self).__init__()

        pf_dim = hid_dim * 4
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.tanh(self.fc_1(x)))
        x = self.fc_2(x)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=0):
        super(AttentionBlock, self).__init__()

        self.attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _x = self.attn_layer_norm(x)
        _x = self.attention(_x, _x, _x)
        x = x + self.dropout(_x)
        x = x + self.dropout(_x)
        _x = self.positionwise_feedforward(self.ff_layer_norm(x))
        x = x + self.dropout(_x)

        return x


class SSVEP_NN(torch.nn.Module):
    def __init__(self, input_len=1000, input_channels=64, output_len=40):
        super(SSVEP_NN, self).__init__()
        kernel_size = 1
        n_heads = 2
        self.input_len = input_len
        self.input_channels = input_channels
        self.output_len = output_len

        self.comb = nn.Sequential(
            nn.Conv1d(self.input_channels, self.input_channels // 2, kernel_size, padding=kernel_size // 2),
            nn.Conv1d(self.input_channels // 2, self.input_channels // 4, kernel_size, padding=kernel_size // 2),
            nn.Conv1d(self.input_channels // 4, 1, kernel_size, padding=kernel_size // 2),
            nn.Dropout(0.2))
        self.attn = nn.Sequential(AttentionBlock(input_len, n_heads),AttentionBlock(input_len, n_heads))
        self.out = nn.Sequential(nn.Dropout(0.2), nn.ReLU(), nn.Linear(input_len, output_len))

    def forward(self, x):
        x = self.comb(x)
        x = self.attn(x)
        x = self.out(x)
        x = x.view(-1, x.size(2))

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
        model = SSVEP_NN(1000, 64, 40)

    device = torch.device("cuda")
    model = model.to(device)
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = op.Adam(model.parameters(), lr=1e-4)

    train_epochs = 10
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
