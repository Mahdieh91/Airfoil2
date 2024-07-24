# 翼型形状隐变量作为条件，预测翼型形状的CVAE
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.nn import init
from torch.utils.data import DataLoader
# from cpAE20 import Autoencoder

# 定义模型
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        # encoder
        self.fc1 = nn.Linear(219, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc31 = nn.Linear(64, 4)
        self.fc32 = nn.Linear(64, 4)

        # decoder
        self.fc4 = nn.Linear(24, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, 199)

    def encoder(self, x, y):
        x = torch.cat((x, y), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc31(x)
        logvar = self.fc32(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decoder(self, z, y):
        z = torch.cat((z, y), 1)
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        x_recon = self.fc6(z)
        return x_recon

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, y)
        return recon, mu, logvar


# 定义损失函数
def loss_function(recon, x, mu, logvar):
    # 重建误差（MSE损失）
    mse_loss = F.mse_loss(recon, x, reduction='sum')
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse_loss + kl_loss


def tain_epoch(train_dataloader, airfoilcvae, criterion, optimizer):
    airfoilcvae.train()
    for train_dataloader1 in train_dataloader:
        optimizer.zero_grad()
        recon, mu, logvar = airfoilcvae(train_dataloader1[:, :199], train_dataloader1[:, 199:])
        loss = loss_function(recon, train_dataloader1[:, :199], mu, logvar)
        loss.backward(retain_graph=True)
        optimizer.step()
        mse = criterion(recon[:, :199], train_dataloader1[:, :199])
    return mse


def test_epoch(test_data, airfoilcvae, criterion):
    airfoilcvae.eval()
    with torch.no_grad():
        recon, mu, logvar = airfoilcvae(test_data[:, :199], test_data[:, 199:])
        test_mse = criterion(recon[:, :199], test_data[:, :199])
    return test_mse


def main_train():
    # 模型超参数
    lr = 0.001  # 学习率
    num_epochs = 1000

    # 加载数据和运行训练和测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建CVAE模型和优化器
    airfoilcvae = CVAE().to(device)
    autoencoder = torch.load('autoencoder20.pkl').to(device)
    optimizer = optim.Adam(airfoilcvae.parameters(), lr=lr)
    criterion = nn.MSELoss().to(device)

    # 加载数据
    data = np.loadtxt('airfoils_recon_data.dat')
    airfoil = data[:, :199]
    airfoil_tensor = torch.tensor(airfoil, dtype=torch.float32)
    airfoil_tensor = airfoil_tensor.to(device)
    cp_latent = np.loadtxt('cpAE_latent20.dat')
    cp_latent_tensor = torch.tensor(cp_latent, dtype=torch.float32).to(device)
    airfoil_nom_latent_tensor = torch.cat((airfoil_tensor, cp_latent_tensor), 1)

    train_data, test_data = train_test_split(airfoil_nom_latent_tensor, test_size=0.01, random_state=42)
    train_dataloader = DataLoader(train_data, batch_size=512, shuffle=True)

    cvae_train_loss = []
    cvae_test_loss = []
    coe = np.arange(0.1, 1, 0.1)
    for epoch in range(num_epochs):
        if epoch in num_epochs * coe:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        train_loss = tain_epoch(train_dataloader, airfoilcvae, criterion, optimizer)
        test_loss = test_epoch(test_data, airfoilcvae, criterion)
        cvae_train_loss.append(train_loss.cpu().detach().numpy())
        cvae_test_loss.append(test_loss.cpu().detach().numpy())

        if (epoch + 1) % 1 == 0:
            print(f'MLP训练 Epoch [{epoch + 1}/{num_epochs}], CVAE_train_loss: {cvae_train_loss[-1]}, '
                  f'CVAE_test_loss:{cvae_test_loss[-1]}')
        # if (epoch + 1) == 1:
        #     torch.save(airfoilcvae, f'apd_cvae_{epoch + 1}.pkl')
        # if (epoch + 1) % 10 == 0:
        #     torch.save(airfoilcvae, f'apd_cvae_{epoch + 1}.pkl')
    torch.save(airfoilcvae, 'apd_cvae.pkl')
    np.savetxt('apd_cvae_train_loss.dat', cvae_train_loss, delimiter='\t')
    np.savetxt('apd_cvae_test_loss.dat', cvae_test_loss, delimiter='\t')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main_train()
