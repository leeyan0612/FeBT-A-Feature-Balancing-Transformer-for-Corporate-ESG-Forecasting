import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from model.MAE import MAE

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data = data_root

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def loss_function(x_hat, x_label):
    MSE = F.mse_loss(x_hat, x_label)
    return MSE

def main():
    parser = argparse.ArgumentParser(description="Masking Auto-Encoder Example")
    parser.add_argument('--batch_size', type=int, default=50, metavar='N', help='batch size for training(default: 1)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train(default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed(default: 1)')
    parser.add_argument('--test_every', type=int, default=5, metavar='N', help='test after every epochs')
    parser.add_argument('--num_worker', type=int, default=1, metavar='N', help='the number of workers')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate(default: 0.001)')
    args = parser.parse_args()

    # 加载数据集
    dataset = pd.read_csv("processed_data_mask.txt", sep='\s', header=None)

    # 数据转换为Tensor
    source_data = torch.tensor(np.array(dataset)).to(torch.float32)

    # 创建 DataLoader
    torch_data = GetLoader(source_data)
    train_batchsize = 8588
    ce_train = DataLoader(torch_data, train_batchsize, shuffle=False, drop_last=True)

    # 模型初始化和配置
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.cuda.manual_seed(100) if cuda else torch.manual_seed(100)

    n = 0
    model = VAE(n, args.epochs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练过程
    loss_list_train = []
    loss_epoch_train = []
    start_epoch = 0
    all_x_label, all_x_hat, all_z_final = [], [], []

    for epoch in range(start_epoch, args.epochs):
        loss_batch_train = []
        n += 1

        for batch_index, x in enumerate(ce_train):
            x = x.to(device)
            model.train()
            x_hat, z, x_label, z_final = model(x, n, args.epochs)
            loss = loss_function(x_hat, x_label)
            loss_batch_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 添加打印每个批次的损失
            if (batch_index + 1) % args.test_every == 0:
                print(
                    f'Epoch: {epoch + 1}/{args.epochs}, Batch: {batch_index + 1}/{len(ce_train)}, Loss: {loss.item()}')

        # 计算并打印每个epoch的平均损失
        epoch_loss_avg = np.sum(loss_batch_train) / len(ce_train)
        loss_epoch_train.append(epoch_loss_avg)
        print(f'End of Epoch {epoch + 1}/{args.epochs}, Average Loss: {epoch_loss_avg}')

        # 在训练过程中收集这些数据
        all_x_label.append(x_label.cpu().detach().numpy())
        all_x_hat.append(x_hat.cpu().detach().numpy())
        all_z_final.append(z_final.cpu().detach().numpy())

    # 将收集的数据进行合并
    all_x_label = np.concatenate(all_x_label, axis=0)
    all_x_hat = np.concatenate(all_x_hat, axis=0)
    all_z_final = np.concatenate(all_z_final, axis=0)

    np.savetxt("processed_data_z_final.txt", all_z_final, encoding='utf-8')
    loss_list_train.append(loss_epoch_train)

if __name__ == '__main__':
    main()
