import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 定义CVAE类
class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_sizes, latent_dim, activation_function):
        super(CVAE, self).__init__()
        self.activation_function = activation_function

        # Encoder layers
        self.encoders = nn.ModuleList()
        in_dim = input_dim + 2
        for h_dim in hidden_sizes:
            self.encoders.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        self.fc_mu = nn.Linear(hidden_sizes[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_sizes[-1], latent_dim)

        # Decoder layers
        self.decoders = nn.ModuleList()
        in_dim = latent_dim + 2
        for h_dim in reversed(hidden_sizes):
            self.decoders.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        self.fc_out = nn.Linear(hidden_sizes[0], input_dim)

    def encoder(self, x, condition):
        x = torch.cat((x, condition), dim=1)
        for layer in self.encoders:
            x = layer(x)
            x = self.get_activation(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z, condition):
        z = torch.cat((z, condition), dim=1)
        for layer in self.decoders:
            z = layer(z)
            z = self.get_activation(z)
        return self.fc_out(z)

    def forward(self, x, condition):
        mu, logvar = self.encoder(x, condition)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, condition), mu, logvar

    def get_activation(self, x):
        if self.activation_function == 'relu':
            return F.relu(x)
        elif self.activation_function == 'tanh':
            return torch.tanh(x)
        elif self.activation_function == 'leaky_relu':
            return F.leaky_relu(x)
        elif self.activation_function == 'sigmoid':
            return torch.sigmoid(x)
        else:
            raise ValueError("Invalid activation function")


def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return 2*MSE + KLD, MSE, KLD


def random_search(dataset, input_dim, device, n_iter=20, epochs=1000):
    best_params = None
    best_loss = float('inf')

    for i in range(n_iter):
        # # 随机选择超参数
        # hidden_layers = np.random.choice(range(2, 6))
        # hidden_sizes = [np.random.choice([50, 100, 150, 200, 250]) for _ in range(hidden_layers)]
        # learning_rate = np.random.choice([1e-5, 1e-4, 1e-3, 1e-2])
        # latent_dim = np.random.choice([2, 4, 8, 16, 32])
        # batch_size = np.random.choice([32, 64, 128, 256, 512, 1024, 2048])
        # # activation_function = np.random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu'])
        # activation_function = np.random.choice(['relu'])

        hidden_layers = 2
        hidden_sizes = [200, 100]
        learning_rate = 0.001
        latent_dim = 4
        activation_function = 'relu'
        # batch_size = [32, 64, 128, 256, 512, 1024, 2048]
        batch_size = [512]
        batch_size = batch_size[i]

        # 打印调试信息
        print(f'Iteration [{i + 1}/{n_iter}], Hidden sizes: {hidden_sizes}, Learning rate: {learning_rate}, Latent dim: {latent_dim}, Batch size: {batch_size}, Activation function: {activation_function}')

        # 创建模型
        cvae = CVAE(input_dim=input_dim, hidden_sizes=hidden_sizes, latent_dim=latent_dim,
                  activation_function=activation_function).to(device)
        optimizer = optim.Adam(cvae.parameters(), lr=learning_rate)

        # 创建训练集和测试集
        train_data, test_data = train_test_split(dataset, test_size=0.01, random_state=42)

        # 创建训练DataLoader
        batch_size = int(batch_size)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # 训练模型
        cvae_train_loss = []
        cvae_test_loss = []
        for epoch in range(epochs):
            cvae.train()
            for train_batch_data in train_loader:
                optimizer.zero_grad()
                recon, mu, logvar = cvae(train_batch_data[:, :199], train_batch_data[:, 199:201])
                loss, mse_loss, kld_loss = loss_function(recon, train_batch_data[:, :199], mu, logvar)
                loss.backward()
                optimizer.step()

            recon, mu, logvar = cvae(train_data[:, :199], train_data[:, 199:201])
            train_loss = F.mse_loss(recon, train_data[:, :199])
            cvae_train_loss.append(train_loss.item())

            # 验证模型
            cvae.eval()
            with torch.no_grad():
                test_recon, mu, logvar = cvae(test_data[:, :199], test_data[:, 199:201])
                test_loss = F.mse_loss(test_recon, test_data[:, :199])
                cvae_test_loss.append(test_loss.item())

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{epochs}], train_loss: {train_loss.item():.8f}, test_loss: {test_loss.item():.8f}')

            # 保存模型
            torch.save(cvae.state_dict(), 'best_cvae.pth')
            np.savetxt('cvae_train_loss.dat', cvae_train_loss, delimiter='\t')
            np.savetxt('cvae_test_loss.dat', cvae_test_loss, delimiter='\t')

    return best_params, best_loss


def main_train():
    # 准备训练数据
    data = np.loadtxt('airfoils_recon_data.dat')
    # 剔除样本太少的数据
    data = data[(data[:, 199] > 0.02) & (data[:, 199] < 0.3)]  # thickness
    data = data[(data[:, 200] > 0.5) & (data[:, 200] < 1.5)]  # cl
    data = data[(data[:, 201] > 0.01)]  # cd
    data = data[(data[:, 202] > 15) & (data[:, 202] < 65)]  # ld

    airfoil_condition = data[:, :201]
    airfoil_condition_min = np.min(airfoil_condition, axis=0)
    airfoil_condition_max = np.max(airfoil_condition, axis=0)
    airfoil_condition_nom = (airfoil_condition - airfoil_condition_min) / (airfoil_condition_max - airfoil_condition_min)
    dataset = torch.FloatTensor(airfoil_condition_nom).to(device)

    # 执行随机搜索
    best_params, best_loss = random_search(dataset, input_dim=199, device=device, n_iter=1, epochs=1000)
    print("Best parameters:", best_params)
    print("Best loss:", best_loss)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_train()
