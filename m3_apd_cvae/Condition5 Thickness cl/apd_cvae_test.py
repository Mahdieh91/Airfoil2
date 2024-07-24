import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from apd_cvae_train import CVAE, loss_function
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
import subprocess
import matplotlib.ticker as ticker

# 准备训练数据
data = np.loadtxt('airfoils_recon_data.dat')
# 剔除样本太少的数据
data = data[(data[:, 199] > 0.02) & (data[:, 199] < 0.3)]  # thickness
data = data[(data[:, 200] > 0.5) & (data[:, 200] < 1.5)]  # cl
data = data[(data[:, 201] > 0.01)]  # cd
data = data[(data[:, 202] > 15) & (data[:, 202] < 65)]  # ld

airfoil = data[:, :199]
airfoil_min = np.min(airfoil, axis=0)
airfoil_max = np.max(airfoil, axis=0)

airfoil_condition = data[:, :201]
airfoil_condition_min = np.min(airfoil_condition, axis=0)
airfoil_condition_max = np.max(airfoil_condition, axis=0)
airfoil_condition_nom = (airfoil_condition - airfoil_condition_min) / (airfoil_condition_max - airfoil_condition_min)
dataset = torch.FloatTensor(airfoil_condition_nom)

# 创建训练集和测试集
train_data, test_data = train_test_split(dataset, test_size=0.01, random_state=42)

# 最佳超参数
best_params = ([200, 100], 0.001, 4, 'relu', 512)
hidden_sizes, learning_rate, latent_dim, activation_function, batch_size = best_params

cvae = CVAE(input_dim=199, hidden_sizes=hidden_sizes, latent_dim=latent_dim, activation_function=activation_function)
cvae.load_state_dict(torch.load('best_cvae.pth'))

# # 0 plot loss
# train_loss20 = np.loadtxt('cvae_train_loss.dat')
# test_loss20 = np.loadtxt('cvae_test_loss.dat')
# plt.figure(figsize=(4, 3.5), dpi=300)
# plt.rcParams["font.family"] = "Times New Roman"
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# plt.semilogy(np.arange(1000), test_loss20, linewidth=1, color='red', label='Train')
# plt.semilogy(np.arange(1000), train_loss20, linewidth=1, color='blue', label='Test')
# plt.ylim(0.0001, 0.1)
# plt.tight_layout()
# plt.legend(frameon=False)
# output_path = r'E:\D_PHD\D6_Project\pre_cp\AirfoilVAE\image\apd_cvae_thick_cl_loss.png'
# plt.savefig(output_path, dpi=300)
# plt.show()

# # 0 训练集
# start_time = time.time()
# cvae.eval()
# recon, mu, logvar = cvae(train_data[:, :199], train_data[:, 199:201])
# airfoils_recon = recon.detach().numpy()
# airfoils_recon = airfoils_recon * (airfoil_max - airfoil_min) + airfoil_min
# train_airfoil = train_data[:, :199].detach().numpy()
# train_airfoil = train_airfoil * (airfoil_max - airfoil_min) + airfoil_min
# airfoil_mse = np.mean((airfoils_recon - train_airfoil) ** 2)
# print('airfoil_mse：', airfoil_mse)
# end_time = time.time()
# elapsed_time = end_time - start_time
# print('elapsed_time', elapsed_time)

# # 1 测试集
# start_time = time.time()
# cvae.eval()
# recon, mu, logvar = cvae(test_data[:, :199], test_data[:, 199:201])
# airfoils_recon = recon.detach().numpy()
# airfoils_recon = airfoils_recon * (airfoil_max - airfoil_min) + airfoil_min
# test_airfoil = test_data[:, :199].detach().numpy()
# test_airfoil = test_airfoil * (airfoil_max - airfoil_min) + airfoil_min
# airfoil_mse = np.mean((airfoils_recon - test_airfoil) ** 2)
# print('airfoil_mse：', airfoil_mse)
# end_time = time.time()
# elapsed_time = end_time - start_time
# print('elapsed_time', elapsed_time)
# mse = np.mean((airfoils_recon - test_airfoil) ** 2, axis=1)
#
# mse = mse*1e6
# # statistic
# plt.figure(figsize=(4, 4), dpi=300)
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 12})
# plt.hist(mse, bins=np.linspace(0, 1, 51), color='g', alpha=0.75, edgecolor='black')
# plt.xlabel(r'MSE($\times10^{-6}$)', fontsize=12)
# plt.ylabel('Count', fontsize=12)
# # plt.xlim(0.02, 0.3)
# # plt.ylim(0, 20000)
# ax = plt.gca()
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
# plt.tight_layout()
# output_path = r'E:\D_PHD\D6_Project\pre_cp\VAEAirfoil_EAAI\fig\apd_cvae_thick_cl_statistic_test1.png'
# plt.savefig(output_path, dpi=300)
# plt.show()
#
# # 2 naca airfoil
# start_time = time.time()
# naca_data = np.loadtxt('naca_data.dat')[:, :201]
# naca_data = (naca_data - airfoil_condition_min) / (airfoil_condition_max - airfoil_condition_min)
# naca_data = torch.FloatTensor(naca_data)
# cvae.eval()
# recon, mu, logvar = cvae(naca_data[:, :199], naca_data[:, 199:201])
# airfoils_recon = recon.detach().numpy()
# airfoils_recon = airfoils_recon * (airfoil_max - airfoil_min) + airfoil_min
# test_airfoil = naca_data[:, :199].detach().numpy()
# test_airfoil = test_airfoil * (airfoil_max - airfoil_min) + airfoil_min
# airfoil_mse = np.mean((airfoils_recon - test_airfoil) ** 2)
# print('airfoil_mse：', airfoil_mse)
# end_time = time.time()
# elapsed_time = end_time - start_time
# print('elapsed_time', elapsed_time)

# show compare test_data2
airfoils2_recon_data = np.loadtxt('airfoils2_recon_data.dat')[:, :201]
airfoils2_recon_data = (airfoils2_recon_data - airfoil_condition_min) / (airfoil_condition_max - airfoil_condition_min)
airfoils2_recon_data = torch.FloatTensor(airfoils2_recon_data)
cvae.eval()
recon, mu, logvar = cvae(airfoils2_recon_data[:, :199], airfoils2_recon_data[:, 199:201])
airfoils2_recon_recon = recon.detach().numpy()
airfoils2_recon_recon = airfoils2_recon_recon * (airfoil_max - airfoil_min) + airfoil_min
np.savetxt('airfoils2_recon_recon.dat', airfoils2_recon_recon, delimiter='\t')

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
# # 生成误差分布图
# plt.Figure(figsize=(4, 4), dpi=300)
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 20})
# plt.title('Condition1 Thickness')
# plt.plot([0, 0.4], [0, 0.4], color='black', linewidth=1)
# plt.scatter(test_data[:, 199].numpy(), airfoils_recon_thickness, s=8, color='g')
# plt.xlim(0, 0.4)
# plt.ylim(0, 0.4)
# plt.xlabel('Specified thickness')
# plt.ylabel('Reconstructed thickness')
# # plt.axis('equal')
# plt.tight_layout()
# plt.show()

# 2 生成1000指定厚度升力系数的翼型
# cvae.eval()
# airfoils_recon = []
# thick_list = np.linspace(0.02, 0.3, 100)
# thick_list = (thick_list - airfoil_condition_min[199]) / (airfoil_condition_max[199] - airfoil_condition_min[199])
# cl_list = np.linspace(0.5, 1.5, 100)
# cl_list = (cl_list - airfoil_condition_min[200]) / (airfoil_condition_max[200] - airfoil_condition_min[200])
#
# for i in range(100):
#     thick = np.random.choice(thick_list, size=(1,))
#     thick = torch.tensor(thick, dtype=torch.float32)
#     cl = np.random.choice(cl_list, size=(1,))
#     cl = torch.tensor(cl, dtype=torch.float32)
#     conditon = torch.cat((thick, cl), dim=0).unsqueeze(0)
#     sample_latent = torch.FloatTensor(np.random.normal(loc=0, scale=1, size=(1, latent_dim)))
#     generated_sample = cvae.decoder(sample_latent, conditon)
#     generated_sample = torch.cat((generated_sample, conditon), dim=1)
#     if len(airfoils_recon) == 0:
#         airfoils_recon = generated_sample.detach().numpy()
#     else:
#         airfoils_recon = np.vstack((airfoils_recon, generated_sample.detach().numpy()))
#
# airfoils_recon = airfoils_recon*(airfoil_condition_max - airfoil_condition_min)+airfoil_condition_min
# np.savetxt('airfoils_thick_cl_recon.dat', airfoils_recon, fmt='%0.6f')
#
# time.sleep(1)
# subprocess.run(["python", "1getData.py"])
# time.sleep(4)

recon_data = np.loadtxt('airfoils_thick_ld_recon_data.dat')
recon_data_airfoil = recon_data[:, :199]
selected_thick = recon_data[:, 199]
rencon_thick = recon_data[:, 200]
selected_ld = recon_data[:, 201]
rencon_ld = recon_data[:, 202]
recon_mse_1 = np.mean((selected_thick - rencon_thick) ** 2)
recon_mse_2 =np.mean((selected_ld - rencon_ld) ** 2)
recon_mse = (recon_mse_1 + recon_mse_2)/2
print('recon_mse：', recon_mse)

# 计算r2
y_true = selected_thick
y_pred = rencon_thick
mse = np.mean((y_true - y_pred) ** 2)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2_1 = 1 - (ss_res / ss_tot)

y_true = selected_ld
y_pred = rencon_ld
mse = np.mean((y_true - y_pred) ** 2)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2_2 = 1 - (ss_res / ss_tot)
print("(R^2):", (r2_1 + r2_2)/2)

# 计算 MAE
mae_1 = np.mean(np.abs(selected_thick - rencon_thick))
mae_2 = np.mean(np.abs(selected_ld - rencon_ld))
print("MAE:", (mae_1 + mae_2)/2)
print()

# 展示重建翼型
loc_x = np.loadtxt('loc_x.dat')
plt.figure(figsize=(4, 3), dpi=300)
plt.rcParams["font.family"] = "Times New Roman"
plt.ylabel('y')
plt.xlabel('x/c')
for i in range(len(recon_data)):
# for i in range(500, 560):
    plt.plot(loc_x, recon_data[i, :199], linewidth=0.5)
plt.axis('equal')
plt.tight_layout()
plt.show()

# 设置画布大小和分辨率
plt.figure(figsize=(4, 3.5), dpi=300)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.plot([0.05, 0.3], [0.05, 0.3], color='black', linewidth=1)
plt.scatter(selected_thick, rencon_thick, s=8, color='g')
plt.xlim(0.05, 0.3)
plt.ylim(0.05, 0.3)
plt.xlabel('Specified thickness', fontsize=12)
plt.ylabel('Reconstructed thickness', fontsize=12)
plt.tight_layout()
output_path = r'E:\D_PHD\D6_Project\pre_cp\AirfoilVAE\image\apd_cvae_thickness_accuracy.png'
plt.savefig(output_path, dpi=300)
plt.show()

# 设置画布大小和分辨率
plt.figure(figsize=(4, 3.5), dpi=300)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.plot([0.5, 1.5], [0.5, 1.5], color='black', linewidth=1)
plt.scatter(selected_ld, rencon_ld, s=8, color='g')
plt.xlim(0.5, 1.5)
plt.ylim(0.5, 1.5)
plt.xlabel('Specified thickness', fontsize=12)
plt.ylabel('Reconstructed thickness', fontsize=12)
plt.tight_layout()
output_path = r'E:\D_PHD\D6_Project\pre_cp\AirfoilVAE\image\apd_cvae_cl_accuracy.png'
plt.savefig(output_path, dpi=300)
plt.show()

# # 2 生成100个厚度为0.15的翼型
# target_thickness = torch.FloatTensor([[0.15]])
# target_thickness = target_thickness.repeat(100, 1, 1)
# sample_mu = np.random.normal(loc=0, scale=1.0, size=(100, latent_dim))
# sample_mu = torch.tensor(sample_mu, dtype=torch.float32)
#
# recon = cvae.decoder(sample_mu, target_thickness.view(-1, 1))
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
# output_path = r'E:\D_PHD\D6_Project\pre_cp\AirfoilVAE\image\airfiol_cvae_thickness_airfoil.png'
# plt.savefig(output_path, dpi=300)
# plt.show()
#
# # # 3 生成误差分布图
# # plt.Figure(figsize=(3.5, 4), dpi=300)
# # plt.rcParams["font.family"] = "Times New Roman"
# # plt.rcParams.update({'font.size': 20})
# # plt.title('Condition1 Thickness')
# # plt.plot([0, 0.5], [0, 0.5], color='black', linewidth=1)
# # plt.scatter(selected_thickness.numpy(), airfoils_recon_thickness, s=8, color='g')
# # plt.xlim(0, 0.5)
# # plt.ylim(0, 0.5)
# # plt.xlabel('Specified thickness')
# # plt.ylabel('Reconstructed thickness', fontdict={'fontname': 'Times New Roman', 'fontsize': 12})
# # # plt.axis('equal')
# # plt.tight_layout()
# # plt.show()
#
# # 设置画布大小和分辨率
# plt.figure(figsize=(4, 3.5), dpi=300)
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 12})  # 调整为更小的字体大小
# plt.plot([0.05, 0.3], [0.05, 0.3], color='black', linewidth=1)
# plt.scatter(selected_thickness.numpy(), airfoils_recon_thickness, s=8, color='g')
# plt.xlim(0.05, 0.3)
# plt.ylim(0.05, 0.3)
# plt.xlabel('Specified thickness', fontsize=12)
# plt.ylabel('Reconstructed thickness', fontsize=12)
# plt.tight_layout()
# output_path = r'E:\D_PHD\D6_Project\pre_cp\AirfoilVAE\image\airfiol_cvae_thickness_accuracy.png'
# plt.savefig(output_path, dpi=300)
# plt.show()
