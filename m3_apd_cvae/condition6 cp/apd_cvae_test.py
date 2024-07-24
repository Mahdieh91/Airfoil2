# 翼型形状隐变量作为条件，预测翼型形状的CVAE
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker
from torch.nn import init
from torch.utils.data import DataLoader
from cpAE20 import Autoencoder
from apd_cvae_train import CVAE
import time

apd_cvae = torch.load('apd_cvae.pkl', map_location=torch.device('cpu'))
criterion = nn.MSELoss()  # 均方误差损失
autoencoder = torch.load('autoencoder20.pkl', map_location=torch.device('cpu'))

# 加载数据
data = np.loadtxt('airfoils_recon_data.dat')
airfoil = data[:, :199]
airfoil_tensor = torch.tensor(airfoil, dtype=torch.float32)
cp_latent = np.loadtxt('cpAE_latent20.dat')
cp_latent_tensor = torch.tensor(cp_latent, dtype=torch.float32)
airfoil_nom_latent_tensor = torch.cat((airfoil_tensor, cp_latent_tensor), 1)

train_data, test_data = train_test_split(airfoil_nom_latent_tensor, test_size=0.01, random_state=42)

# # 0 plot loss
# train_loss20 = np.loadtxt('apd_cvae_train_loss.dat')
# test_loss20 = np.loadtxt('apd_cvae_test_loss.dat')
# plt.figure(figsize=(4, 3.5), dpi=300)
# plt.rcParams["font.family"] = "Times New Roman"
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# plt.semilogy(np.arange(1000), test_loss20, linewidth=1, color='red', label='Train')
# plt.semilogy(np.arange(1000), train_loss20, linewidth=1, color='blue', label='Test')
# plt.ylim(0.000000001, 0.0001)
# plt.tight_layout()
# plt.legend(frameon=False)
# output_path = r'E:\D_PHD\D6_Project\pre_cp\VAEAirfoil_EAAI\fig\apd_cvae_pressure_loss.png'
# plt.savefig(output_path, dpi=300)
# plt.show()

# # cvae train set
# start_time = time.time()
# recon, mu, logvar = apd_cvae(train_data[:, :199], train_data[:, 199:])
# train_MSE = criterion(recon[:, :199], train_data[:, :199])
# print(f'train loss = {train_MSE}')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print('elapsed_time', elapsed_time)
#
# # cvae test set
# recon, mu, logvar = apd_cvae(test_data[:, :199], test_data[:, 199:])
# test_MSE = criterion(recon[:, :199], test_data[:, :199])
# print(f'test loss = {test_MSE}')
# mse = np.mean((recon[:, :199].detach().numpy() - test_data[:, :199].detach().numpy()) ** 2, axis=1)
# mse = mse*1e8
# # statistic
# plt.figure(figsize=(4, 4), dpi=300)
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 12})
# plt.hist(mse, bins=np.linspace(0, 5, 51), color='g', alpha=0.75, edgecolor='black')
# plt.xlabel(r'MSE($\times10^{-8}$)', fontsize=12)
# plt.ylabel('Count', fontsize=12)
# # plt.xlim(0.02, 0.3)
# # plt.ylim(0, 20000)
# ax = plt.gca()
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
# plt.tight_layout()
# output_path = r'E:\D_PHD\D6_Project\pre_cp\VAEAirfoil_EAAI\fig\apd_cvae_cp_statistic_test1.png'
# plt.savefig(output_path, dpi=300)
# plt.show()
#
# # naca data set
# data = np.loadtxt('naca_data.dat')
# airfoil = data[:, :199]
# airfoil_tensor = torch.tensor(airfoil, dtype=torch.float32)
# cp_latent = np.loadtxt('naca_cpAE_latent20.dat')
# cp_latent_tensor = torch.tensor(cp_latent, dtype=torch.float32)
# airfoil_nom_latent_tensor = torch.cat((airfoil_tensor, cp_latent_tensor), 1)
# recon, mu, logvar = apd_cvae(airfoil_nom_latent_tensor[:, :199], airfoil_nom_latent_tensor[:, 199:])
# naca_mse = criterion(recon[:, :199], airfoil_nom_latent_tensor[:, :199])
# print(f'naca loss = {naca_mse}')
# mse = np.mean((recon[:, :199].detach().numpy() - airfoil_nom_latent_tensor[:, :199].detach().numpy()) ** 2, axis=1)
# # statistic
# plt.figure(figsize=(4, 4), dpi=300)
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 12})
# plt.hist(mse, bins=50, color='g', alpha=0.75, edgecolor='black')
# plt.xlabel('MSE', fontsize=12)
# plt.ylabel('Count', fontsize=12)
# # plt.xlim(0.02, 0.3)
# # plt.ylim(0, 20000)
# ax = plt.gca()
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
# plt.tight_layout()
# output_path = r'E:\D_PHD\D6_Project\pre_cp\VAEAirfoil_EAAI\fig\apd_cvae_cp_statistic_naca.png'
# plt.savefig(output_path, dpi=300)
# plt.show()

# show compare test_data2
data = np.loadtxt('airfoils2_recon_data.dat')
airfoil = data[:, :199]
airfoil_tensor = torch.tensor(airfoil, dtype=torch.float32)
cp_latent = np.loadtxt('airfoils2_recon_data_latent20.dat')
cp_latent_tensor = torch.tensor(cp_latent, dtype=torch.float32)
airfoil_nom_latent_tensor = torch.cat((airfoil_tensor, cp_latent_tensor), 1)
recon, mu, logvar = apd_cvae(airfoil_nom_latent_tensor[:, :199], airfoil_nom_latent_tensor[:, 199:])
recon = recon.detach().numpy()
np.savetxt('airfoils2_recon_recon.dat', recon, delimiter='\t')


# loc_x = np.loadtxt('loc_x.dat')
# plt.Figure()
# plt.title('airfoilCVAE')
# for i in range(40, 41):
#     plt.scatter(loc_x, train_data[i, :199].detach().numpy(), s=2, color='blue')
#     plt.plot(loc_x, recon[i, :199].detach().numpy(), linewidth=1, color='red')
# plt.axis('equal')
# plt.show()


# 生成多个翼型对比图
loc_x = np.loadtxt('loc_x.dat')
fig, axes = plt.subplots(4, 4, figsize=(16, 16), gridspec_kw={'wspace': 0.05, 'hspace': -0.85}, dpi=300)
# fig, axes = plt.subplots(3, 4, figsize=(16, 16))
numbers = list(range(1, len(test_data)))
random.seed(42)
rand_num = random.sample(numbers, 16)
for i in range(16):
    # axes[i].set_xlabel('x/c', fontdict={'style': 'italic'})
    # axes[i].set_ylabel('y', fontdict={'style': 'italic'})
    row = i // 4
    col = i % 4
    axes[row, col].scatter(loc_x, test_data[rand_num[i], :199], s=4, color='red', label='Ground Truth')
    axes[row, col].plot(loc_x, recon[rand_num[i], :].detach().numpy(), linewidth=1.5, color='blue', label='Reconstructed')
    axes[row, col].set_ylim(-0.2, 0.2)
    axes[row, col].set_aspect('equal')
    axes[row, col].set_xticks([])
    axes[row, col].set_yticks([])
    axes[row, col].set_xticklabels([])
    axes[row, col].set_yticklabels([])
    # axes[row, col].invert_yaxis()
plt.tight_layout()
plt.legend(frameon=False)
plt.rcParams["font.family"] = "Times New Roman"
plt.show()

# 随机生成隐变量，并plot误差直方图
sample_mu = np.random.normal(loc=0, scale=1, size=(len(data), 5))
sample_mu = torch.tensor(sample_mu, dtype=torch.float32)
cp_latent = np.loadtxt('cpAE_latent20.dat')
cp_latent_tensor = torch.tensor(cp_latent, dtype=torch.float32)
# airfoil_nom_latent_tensor = torch.cat((airfoil_tensor, cp_latent_tensor), 1)
recon = apd_cvae.decoder(sample_mu, cp_latent_tensor)
recon_MSE = np.mean((airfoil - recon.detach().numpy()) ** 2, axis=1)
#
# bin_edges = np.linspace(np.min(recon_MSE*1e8), 3, num=50)
# hist, _ = np.histogram(recon_MSE*1e8, bins=bin_edges)
# plt.figure(figsize=(4, 3), dpi=300)
# plt.hist(recon_MSE*1e8, bins=bin_edges, edgecolor='k', color='g')
# plt.xlabel(r'MSE($\times 10^{-8}$)')
# plt.ylabel('Count')
# plt.rcParams["font.family"] = "Times New Roman"
# plt.tight_layout()
# plt.show()


# 生成不同epoch下多个翼型对比图
loc_x = np.loadtxt('loc_x.dat')
fig, axes = plt.subplots(4, 1, figsize=(4, 16), gridspec_kw={'wspace': 0.05, 'hspace': -0.85}, dpi=300)
plt.rcParams["font.family"] = "Times New Roman"
# fig, axes = plt.subplots(3, 4, figsize=(16, 16))
numbers = list(range(1, len(test_data)))
random.seed(42)
rand_num = random.sample(numbers, 4)
for i in range(4):
    # axes[i].set_xlabel('x/c', fontdict={'style': 'italic'})
    # axes[i].set_ylabel('y', fontdict={'style': 'italic'})
    axes[i].scatter(loc_x[::2], airfoil[rand_num[i], :199][::2], s=4, color='red', label='Ground Truth')
    axes[i].plot(loc_x, recon[rand_num[i], :].detach().numpy(), linewidth=1.5, color='blue', label='Reconstructed')
    axes[i].set_ylim(-0.2, 0.2)
    axes[i].set_aspect('equal')
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_xticklabels([])
    axes[i].set_yticklabels([])
plt.tight_layout()
# axes[0].set_title('Epoch=1')
plt.legend(frameon=False)
plt.show()


