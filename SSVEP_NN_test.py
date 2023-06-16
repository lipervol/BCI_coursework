from SSVEP_NN import *
import torch
from torch.utils.data import DataLoader
import numpy as np


class test_Dataset(Dataset):
    def __init__(self, path, ID, class_num=40):
        super(test_Dataset, self).__init__()
        self.path = path
        self.ID = ID
        self.class_num = class_num
        self.data = np.load(self.path + "/S%d.npy" % ID)

    def __getitem__(self, idx):
        i, j = idx // self.class_num, idx % self.class_num
        return self.data[i, j], j

    def __len__(self):
        return self.data.shape[0] * self.data.shape[1]


if __name__ == "__main__":

    model_num = 10
    acc_all = []
    device = torch.device("cuda")

    for i in range(model_num):
        model = torch.load(models_save_path + "/model%d.pth" % (i + 1))
        model.to(device)
        test_dataset = test_Dataset(data_path, ID=(i + 1))
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for item in test_dataloader:
                data, label = item
                data = data.type(torch.float32)
                data = data.to(device)
                label = label.type(torch.int64)
                label = label.to(device)

                output = model(data)
                pred = output.argmax(dim=1)

                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += data.size(0)
            acc = total_correct / total_num
            print("[*]model%d\t" % (i + 1) + "Accuracy:%.2f" % acc)
        acc_all.append(acc)
    print("[*]Average accuracy:%.2f" % (np.mean(acc_all)))
