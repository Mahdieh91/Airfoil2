import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from afd_cvae_train import CVAE, loss_function
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import subprocess
import time

# 准备训练数据
data = np.loadtxt('airfoils_recon_data.dat')
loc_x = np.loadtxt('loc_x.dat')
airfoil = data[:, :199]
# 随机噪声，防止坐标为0
noise = np.zeros((len(airfoil), 199))
noise[:, 99] = np.random.normal(0, 1e-5, len(airfoil), )

airfoil_min = np.min(airfoil, axis=0)
airfoil_max = np.max(airfoil, axis=0)
airfoil_nom = (airfoil - airfoil_min) / (airfoil_max - airfoil_min)
airfoil_nom_tensor = torch.FloatTensor(airfoil_nom)
ld = data[:, 202]
ld_nom = (ld - np.min(ld)) / (np.max(ld) - np.min(ld))
ld_tensor = torch.FloatTensor(ld_nom)
dataset = torch.cat((airfoil_nom_tensor, ld_tensor.view(-1, 1)), dim=1)

# 创建训练集和测试集
train_data, test_data = train_test_split(dataset, test_size=0.01, random_state=42)

# 最佳超参数
best_params = ([200, 100], 0.001, 4, 'relu', 512)
hidden_sizes, learning_rate, latent_dim, activation_function, batch_size = best_params

cvae = CVAE(input_dim=199, hidden_sizes=hidden_sizes, latent_dim=latent_dim, activation_function=activation_function)
cvae.load_state_dict(torch.load('best_cvae.pth'))

ld_recon_data = np.loadtxt('airfoils_ld_recon_data.dat')
airfoils_recon = ld_recon_data[:, :199]

# 展示重建翼型
loc_x = np.loadtxt('loc_x.dat')
plt.figure(figsize=(4, 3), dpi=300)
plt.rcParams["font.family"] = "Times New Roman"
plt.ylabel('y')
plt.xlabel('x/c')
for i in range(len(airfoils_recon)):
# for i in range(500, 600):
    plt.plot(loc_x, airfoils_recon[i, :], linewidth=0.5)
plt.axis('equal')
plt.tight_layout()
plt.show()


# 2 生成指定升力系数的翼型
cvae.eval()
selected_ld = np.linspace(0, 1, 1000)
selected_ld = torch.tensor(selected_ld, dtype=torch.float32)
sample_mu = np.random.normal(loc=0, scale=1.0, size=(len(selected_ld), latent_dim))
sample_mu = torch.tensor(sample_mu, dtype=torch.float32)
recon = cvae.decoder(sample_mu, selected_ld.view(-1, 1))
airfoils_recon = recon.detach().numpy()
airfoils_recon = airfoils_recon * (airfoil_max - airfoil_min) + airfoil_min

np.savetxt('airfoils_ld_recon.dat', airfoils_recon, fmt='%0.6f')
time.sleep(1)
subprocess.run(["python", "1getData.py"])
time.sleep(10)
selected_ld = np.linspace(0, 1, 1000)*(np.max(ld) - np.min(ld))+np.min(ld)
ld_recon = np.loadtxt('airfoils_ld_recon.dat')
ld_recon_data = np.loadtxt('airfoils_ld_recon_data.dat')
ld_recon_data_airfoil = ld_recon_data[:, :199]

# 初始化一个列表来存储保留的 selected_ld
filtered_selected_ld = []
# 遍历每一行 ld_recon
for i, ld in enumerate(ld_recon):
    # 检查该行是否在 ld_recon_data 中
    if any((ld_recon_data_airfoil == ld).all(axis=1)):
        # 如果存在，则保留 selected_ld 对应位置的元素
        filtered_selected_ld.append(selected_ld[i])
# 转换为numpy数组
filtered_selected_ld = np.array(filtered_selected_ld)

ld_mse = np.mean((filtered_selected_ld - ld_recon_data[:, 202]) ** 2)
print('ld MSE：', ld_mse)

# 设置画布大小和分辨率
plt.figure(figsize=(4, 3.5), dpi=300)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})  # 调整为更小的字体大小
plt.plot([np.min(filtered_selected_ld), np.max(filtered_selected_ld)], [np.min(filtered_selected_ld), np.max(filtered_selected_ld)], color='black', linewidth=1)
plt.scatter(filtered_selected_ld, ld_recon_data[:, 202], s=8, color='g')
plt.xlim(np.min(filtered_selected_ld), np.max(filtered_selected_ld))
plt.ylim(np.min(filtered_selected_ld), np.max(filtered_selected_ld))
plt.xlabel('Specified ld', fontsize=12)
plt.ylabel('Reconstructed ld', fontsize=12)
plt.tight_layout()
# output_path = r'E:\D_PHD\D6_Project\pre_cp\AirfoilVAE\image\airfiol_cvae_ld_accuracy.png'
# plt.savefig(output_path, dpi=300)
plt.show()



# # 展示重建翼型
# loc_x = np.loadtxt('loc_x.dat')
# plt.figure(figsize=(4, 3), dpi=300)
# plt.rcParams["font.family"] = "Times New Roman"
# plt.ylabel('y')
# plt.xlabel('x/c')
# for i in range(len(airfoils_recon)):
# # for i in range(980, 1000):
#     plt.plot(loc_x, airfoils_recon[i, :], linewidth=0.5)
# plt.axis('equal')
# plt.tight_layout()
# plt.show()
#
# cvae_diversity = np.mean(np.var(airfoils_recon, axis=0))
# airfoil_filter = np.apply_along_axis(lambda x: savgol_filter(x, 15, 3), axis=1, arr=airfoils_recon)
# cvae_roughness = np.mean(np.mean((airfoils_recon - airfoil_filter) ** 2, axis=1))
# print(cvae_diversity/0.00038256164690759424)
# print(cvae_roughness/2.9165800679528876e-09)

# # 2 生成100个厚度为0.15的翼型
# target_ld = torch.FloatTensor([[0.15]])
# target_ld = target_ld.repeat(100, 1, 1)
# sample_mu = np.random.normal(loc=0, scale=1.0, size=(100, latent_dim))
# sample_mu = torch.tensor(sample_mu, dtype=torch.float32)
# 
# recon = cvae.decoder(sample_mu, target_ld.view(-1, 1))
# airfoils_recon = recon.detach().numpy()
# # 反归一化
# airfoils_recon = airfoils_recon * (airfoil_max - airfoil_min) + airfoil_min
# loc_x = np.loadtxt('loc_x.dat')
# # plt.figure(figsize=(4, 3), dpi=300)
# # plt.rcParams["font.family"] = "Times New Roman"
# # plt.ylabel('y')
# # plt.xlabel('x/c')
# # for i in range(len(airfoils_recon)):
# #     plt.plot(loc_x, airfoils_recon[i, :], linewidth=0.5)
# # plt.axis('equal')
# # plt.tight_layout()
# # plt.show()
# 
# # plot三维翼型图
# max_y = np.argmax(airfoils_recon, axis=1)
# sorted_indices = np.lexsort((max_y, airfoils_recon[np.arange(len(airfoils_recon)), max_y]))
# sorted_airfoil = airfoils_recon[sorted_indices][::-1]
# num_airfoils, num_points = sorted_airfoil.shape
# cmap = plt.get_cmap('winter')
# alpha = np.linspace(0.6, 1, 100)[::-1]
# fig = plt.figure(dpi=300)
# ax = fig.add_subplot(111, projection='3d')
# for i in range(num_airfoils):
# # for i in range(5):
#     # x = np.arange(num_airfoils)
#     x = loc_x
#     y = np.full_like(x, i)
#     z = sorted_airfoil[i, :]
#     ax.plot(y, x, z, color=cmap(i/100), alpha=alpha[i])
# # ax.invert_xaxis()
# # ax.set_box_aspect([1, 1, 0.15])
# ax.set_xlabel('Airfoil num', fontsize=12, fontfamily='Times New Roman')
# ax.set_ylabel('x/c', fontsize=12, fontstyle='italic', fontfamily='Times New Roman')
# ax.set_zlabel('y', fontsize=12, fontstyle='italic', fontfamily='Times New Roman')
# ax.set_zlim(-0.4, 0.4)
# plt.rcParams["font.family"] = "Times New Roman"
# plt.tight_layout()
# output_path = r'E:\D_PHD\D6_Project\pre_cp\AirfoilVAE\image\airfiol_cvae_ld_airfoil.png'
# plt.savefig(output_path, dpi=300)
# plt.show()

# # 3 生成误差分布图
# plt.Figure(figsize=(3.5, 4), dpi=300)
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 20})
# plt.title('Condition1 ld')
# plt.plot([0, 0.5], [0, 0.5], color='black', linewidth=1)
# plt.scatter(selected_ld.numpy(), airfoils_recon_ld, s=8, color='g')
# plt.xlim(0, 0.5)
# plt.ylim(0, 0.5)
# plt.xlabel('Specified ld')
# plt.ylabel('Reconstructed ld', fontdict={'fontname': 'Times New Roman', 'fontsize': 12})
# # plt.axis('equal')
# plt.tight_layout()
# plt.show()

# # 设置画布大小和分辨率
# plt.figure(figsize=(4, 3.5), dpi=300)
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 12})  # 调整为更小的字体大小
# plt.plot([np.min(ld), np.max(ld)], [np.min(ld), np.max(ld)], color='black', linewidth=1)
# plt.scatter(selected_ld.numpy(), airfoils_recon_ld, s=8, color='g')
# plt.xlim(np.min(ld), np.max(ld))
# plt.ylim(np.min(ld), np.max(ld))
# plt.xlabel('Specified ld', fontsize=12)
# plt.ylabel('Reconstructed ld', fontsize=12)
# plt.tight_layout()
# output_path = r'E:\D_PHD\D6_Project\pre_cp\AirfoilVAE\image\airfiol_cvae_ld_accuracy.png'
# plt.savefig(output_path, dpi=300)
# plt.show()
