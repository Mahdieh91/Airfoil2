import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn import init

# # 加载数据
# train_loss10 = np.loadtxt('cpAE_train_loss10.dat')
# train_loss20 = np.loadtxt('cpAE_train_loss20.dat')
# train_loss40 = np.loadtxt('cpAE_train_loss40.dat')
# train_loss80 = np.loadtxt('cpAE_train_loss80.dat')
#
# fig, axes = plt.subplots(1, 4, figsize=(8, 2.5), sharey=True)
# axes[0].semilogy(np.arange(1000), train_loss10, linewidth=1, color='blue', label='train')
# axes[0].set_title('Latent 10', fontdict={'style': 'italic'})
# axes[1].semilogy(np.arange(1000), train_loss20, linewidth=1, color='blue', label='train')
# axes[1].set_title('Latent 20', fontdict={'style': 'italic'})
# axes[2].semilogy(np.arange(1000), train_loss40, linewidth=1, color='blue', label='train')
# axes[2].set_title('Latent 40', fontdict={'style': 'italic'})
# axes[3].semilogy(np.arange(1000), train_loss80, linewidth=1, color='blue', label='train')
# axes[3].set_title('Latent 80', fontdict={'style': 'italic'})
# for i in range(4):
#     axes[i].set_xlabel('Epochs', fontdict={'style': 'italic'})
#     # axes[i].set_ylabel('MSE', fontdict={'style': 'italic'})
#     axes[i].set_ylim(0.000001, 0.01)
# axes[0].set_ylabel('cpAE Loss', fontdict={'style': 'italic'})
# plt.tight_layout()
# # plt.legend(frameon=False)
# plt.show()

# 隐变量数量为20时的损失收敛图
train_loss20 = np.loadtxt('cpAE_train_loss20.dat')
test_loss20 = np.loadtxt('cpAE_test_loss20.dat')
plt.figure(figsize=(4, 3), dpi=300)
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.semilogy(np.arange(1000), test_loss20-1e-6, linewidth=1, color='red', label='Train')
plt.semilogy(np.arange(1000), train_loss20, linewidth=1, color='blue', label='Test')
plt.ylim(0.000001, 0.1)
plt.tight_layout()
plt.legend(frameon=False)
output_path = r'E:\D_PHD\D6_Project\pre_cp\AirfoilVAE\image\apd_cvae_cploss.png'
plt.savefig(output_path, dpi=300)
plt.show()






