import random
from glob import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from cpAE20 import Autoencoder

# 加载自编码器
autoencoder = torch.load('autoencoder20.pkl', map_location=torch.device('cpu'))
criterion = nn.MSELoss()  # 均方误差损失

# 加载数据
data = np.loadtxt('airfoils_recon_data.dat')[:, 204:]
# 压力系数归一化
min = np.min(data, axis=0)
max = np.max(data, axis=0)
data_nom = (data - min) / (max - min)

data_nom_tensor = torch.tensor(data_nom, dtype=torch.float32)

_, decoded = autoencoder(data_nom_tensor)
MSE = criterion(decoded, data_nom_tensor)
print('MSE', MSE)

decoded = decoded.detach().numpy()*(max - min) + min

# 生成多个压力分布对比图
loc_x = np.loadtxt('loc_x.dat')
# fig, axes = plt.subplots(3, 4, figsize=(16, 16), gridspec_kw={'wspace': 0.05, 'hspace': -0.67})
fig, axes = plt.subplots(3, 3, figsize=(10, 10), dpi=300)
numbers = list(range(1, len(data_nom_tensor)))
random.seed(43)
rand_num = random.sample(numbers, 12)
for i in range(9):
    # axes[i].set_xlabel('x/c', fontdict={'style': 'italic'})
    # axes[i].set_ylabel('y', fontdict={'style': 'italic'})
    row = i // 3
    col = i % 3
    axes[row, col].scatter(loc_x[::2], data[rand_num[i], :][::2], s=4, color='red', label='Ground Truth')
    axes[row, col].plot(loc_x, decoded[rand_num[i], :], linewidth=1.2, color='blue', label='Reconstructed')
    # axes[row, col].set_ylim(-0.2, 0.2)
    # axes[row, col].set_aspect('equal')
    axes[row, col].set_xticks([])
    axes[row, col].set_yticks([])
    axes[row, col].set_xticklabels([])
    axes[row, col].set_yticklabels([])
    axes[row, col].invert_yaxis()
handles, labels = plt.gca().get_legend_handles_labels()
plt.tight_layout()
plt.rcParams["font.family"] = "Times New Roman"
# plt.legend(handles[::-1], labels[::-1])
plt.legend(frameon=False, prop={'family': 'Times New Roman', 'size': 12})
output_path = r'E:\D_PHD\D6_Project\pre_cp\AirfoilVAE\image\apd_cvae_cp.png'
plt.savefig(output_path, dpi=300)
plt.show()