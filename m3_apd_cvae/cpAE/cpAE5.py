import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn import init

# 超参数设置
from torch.utils.data import DataLoader

learning_rate = 0.001
num_epochs = 1000


# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(199, 140),
            nn.ReLU(),
            nn.Linear(140, 80),
            nn.ReLU(),
            nn.Linear(80, 5)
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 80),
            nn.ReLU(),
            nn.Linear(80, 140),
            nn.ReLU(),
            nn.Linear(140, 199)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def tain_epoch(train_dataloader, autoencoder, criterion, autoencoder_optimizer):
    autoencoder.train()
    total_loss = 0.0
    num_batches = len(train_dataloader)
    for train_dataloader1 in train_dataloader:
        autoencoder_optimizer.zero_grad()
        _, decoded = autoencoder(train_dataloader1)
        autoencoder_loss = criterion(decoded, train_dataloader1)
        autoencoder_loss.backward()
        autoencoder_optimizer.step()
        total_loss += autoencoder_loss.item()
    average_loss = total_loss/num_batches
    return average_loss


def test_epoch(test_dataloader, autoencoder, criterion):
    autoencoder.eval()
    total_loss = 0.0
    num_batches = len(test_dataloader)

    for test_dataloader1 in test_dataloader:
        _, decoded = autoencoder(test_dataloader1)
        autoencoder_loss = criterion(decoded, test_dataloader1)
        total_loss += autoencoder_loss.item()
    average_loss = total_loss / num_batches
    return average_loss


def main_train():
    # 创建自编码器和MLP模型
    autoencoder = Autoencoder()
    autoencoder = autoencoder.cuda()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    criterion = criterion.cuda()
    autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # 加载数据
    data = np.loadtxt('airfoils_recon_data.dat')[:, 204:]
    # 压力系数归一化
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    data_nom = (data - min) / (max - min)

    data_nom_tensor = torch.tensor(data_nom, dtype=torch.float32)
    train_data, test_data = train_test_split(data_nom_tensor, test_size=0.1, random_state=42)
    train_data = train_data.cuda()
    test_data = test_data.cuda()
    train_dataloader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=512, shuffle=True)

    # 自编码器训练
    AE_train_loss = []
    AE_test_loss = []
    coe = np.arange(0.1, 1, 0.2)
    for epoch in range(num_epochs):
        if epoch in num_epochs * coe:
            for param_group in autoencoder_optimizer.param_groups:
                param_group['lr'] *= 0.5

        train_loss = tain_epoch(train_dataloader, autoencoder, criterion, autoencoder_optimizer)
        AE_train_loss.append(train_loss)
        test_loss = test_epoch(test_dataloader, autoencoder, criterion)
        AE_test_loss.append(test_loss)

        if (epoch+1) % 1 == 0:
            print(f'MLP训练 Epoch [{epoch + 1}/{num_epochs}], MLP_train_loss: {AE_train_loss[-1]}, '
                  f'MLP_test_loss:{AE_test_loss[-1]}')

    torch.save(autoencoder, 'autoencoder5.pkl')
    np.savetxt('cpAE_train_loss5.dat', AE_train_loss, delimiter='\t')
    np.savetxt('cpAE_test_loss5.dat', AE_test_loss, delimiter='\t')

    # 自编码器loss图
    plt.Figure()
    plt.title('autoencoder MSE loss')
    plt.xlabel('epochs')
    plt.semilogy(np.arange(num_epochs), AE_train_loss, linewidth=0.5, color='blue', label='train')
    plt.semilogy(np.arange(num_epochs), AE_test_loss, linewidth=0.5, color='red', label='test')
    plt.legend()
    plt.show()

    # # 自编码器验证
    # encoded, decoded = autoencoder(val_data)
    # autoencoder_MSE = criterion(decoded, val_data)
    # print(f'autoencoder_MSE = {autoencoder_MSE}')
    # loc_x = np.loadtxt('loc_x.dat')
    # plt.Figure()
    # plt.title('val_autoencoder')
    # plt.scatter(loc_x, val_data[1, :].detach().numpy(), s=2, color='blue')
    # plt.plot(loc_x, decoded[1, :].detach().numpy(), linewidth=1, color='red')
    # plt.show()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    main_train()


