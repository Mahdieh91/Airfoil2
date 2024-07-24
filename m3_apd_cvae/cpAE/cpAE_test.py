import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn

# from cpAE2 import Autoencoder
# # 加载自编码器
# autoencoder = torch.load('autoencoder2.pkl', map_location=torch.device('cpu'))
# criterion = nn.MSELoss()  # 均方误差损失
#
# # 加载数据
# data = np.loadtxt('airfoils_recon_data.dat')[:, 204:]
# # 压力系数归一化
# min = np.min(data, axis=0)
# max = np.max(data, axis=0)
# data_nom = (data - min) / (max - min)
# data_nom_tensor = torch.tensor(data_nom, dtype=torch.float32)
#
# encoded, decoded = autoencoder(data_nom_tensor)
# autoencoder_MSE = criterion(decoded, data_nom_tensor)
#
# decoded = decoded.detach().numpy()*(max - min) + min
# loc_x = np.loadtxt('loc_x.dat')
# # 生成翼型隐变量数据
# np.savetxt('cpAE_latent2.dat', encoded.detach().numpy(), delimiter='\t')
#
#
# from cpAE5 import Autoencoder
# # 加载自编码器
# autoencoder = torch.load('autoencoder5.pkl', map_location=torch.device('cpu'))
# criterion = nn.MSELoss()  # 均方误差损失
#
# # 加载数据
# data = np.loadtxt('airfoils_recon_data.dat')[:, 204:]
# # 压力系数归一化
# min = np.min(data, axis=0)
# max = np.max(data, axis=0)
# data_nom = (data - min) / (max - min)
# data_nom_tensor = torch.tensor(data_nom, dtype=torch.float32)
#
# encoded, decoded = autoencoder(data_nom_tensor)
# autoencoder_MSE = criterion(decoded, data_nom_tensor)
#
# decoded = decoded.detach().numpy()*(max - min) + min
# loc_x = np.loadtxt('loc_x.dat')
# # 生成翼型隐变量数据
# np.savetxt('cpAE_latent5.dat', encoded.detach().numpy(), delimiter='\t')
#
# from cpAE10 import Autoencoder
# # 加载自编码器
# autoencoder = torch.load('autoencoder10.pkl', map_location=torch.device('cpu'))
# criterion = nn.MSELoss()  # 均方误差损失
#
# # 加载数据
# data = np.loadtxt('airfoils_recon_data.dat')[:, 204:]
# # 压力系数归一化
# min = np.min(data, axis=0)
# max = np.max(data, axis=0)
# data_nom = (data - min) / (max - min)
# data_nom_tensor = torch.tensor(data_nom, dtype=torch.float32)
#
# encoded, decoded = autoencoder(data_nom_tensor)
# autoencoder_MSE = criterion(decoded, data_nom_tensor)
#
# decoded = decoded.detach().numpy()*(max - min) + min
# loc_x = np.loadtxt('loc_x.dat')
# # plt.Figure()
# # plt.title('test_autoencoder')
# # for i in range(5, 6):
# #     plt.plot(loc_x, decoded[i, :], linewidth=2, color='red', label='Predition')
# #     plt.scatter(loc_x, data[i, :], s=5, color='blue', label='Groud Truth')
# # plt.show()
# # 生成翼型隐变量数据
# np.savetxt('cpAE_latent10.dat', encoded.detach().numpy(), delimiter='\t')


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

encoded, decoded = autoencoder(data_nom_tensor)
autoencoder_MSE = criterion(decoded, data_nom_tensor)

decoded = decoded.detach().numpy()*(max - min) + min
loc_x = np.loadtxt('loc_x.dat')
# 生成翼型隐变量数据
np.savetxt('cpAE_latent20.dat', encoded.detach().numpy(), delimiter='\t')

# # 生成naca翼型压力分布的数据
# naca_data = np.loadtxt('naca_data.dat')[:, 204:]
# naca_data_nom = (naca_data - min) / (max - min)
# naca_data_nom_tensor = torch.tensor(naca_data_nom, dtype=torch.float32)
# encoded, decoded = autoencoder(naca_data_nom_tensor)
# autoencoder_MSE = criterion(decoded, naca_data_nom_tensor)
#
# decoded = decoded.detach().numpy()*(max - min) + min
# loc_x = np.loadtxt('loc_x.dat')
# # 生成翼型隐变量数据
# np.savetxt('naca_cpAE_latent20.dat', encoded.detach().numpy(), delimiter='\t')

# 生成对比数据隐变量
airfoils2_recon_data = np.loadtxt('airfoils2_recon_data.dat')[:, 204:]
airfoils2_recon_data_nom = (airfoils2_recon_data - min) / (max - min)
airfoils2_recon_data_nom_tensor = torch.tensor(airfoils2_recon_data_nom, dtype=torch.float32)
encoded, decoded = autoencoder(airfoils2_recon_data_nom_tensor)
autoencoder_MSE = criterion(decoded, airfoils2_recon_data_nom_tensor)
decoded = decoded.detach().numpy()*(max - min) + min
np.savetxt('airfoils2_recon_data_latent20.dat', encoded.detach().numpy(), delimiter='\t')



# from cpAE40 import Autoencoder
# # 加载自编码器
# autoencoder = torch.load('autoencoder40.pkl', map_location=torch.device('cpu'))
# criterion = nn.MSELoss()  # 均方误差损失
#
# # 加载数据
# data = np.loadtxt('airfoils_recon_data.dat')[:, 204:]
# # 压力系数归一化
# min = np.min(data, axis=0)
# max = np.max(data, axis=0)
# data_nom = (data - min) / (max - min)
# data_nom_tensor = torch.tensor(data_nom, dtype=torch.float32)
#
# encoded, decoded = autoencoder(data_nom_tensor)
# autoencoder_MSE = criterion(decoded, data_nom_tensor)
#
# decoded = decoded.detach().numpy()*(max - min) + min
# loc_x = np.loadtxt('loc_x.dat')
# # 生成翼型隐变量数据
# np.savetxt('cpAE_latent40.dat', encoded.detach().numpy(), delimiter='\t')
#
#
#
# from cpAE80 import Autoencoder
# # 加载自编码器
# autoencoder = torch.load('autoencoder80.pkl', map_location=torch.device('cpu'))
# criterion = nn.MSELoss()  # 均方误差损失
#
# # 加载数据
# data = np.loadtxt('airfoils_recon_data.dat')[:, 204:]
# # 压力系数归一化
# min = np.min(data, axis=0)
# max = np.max(data, axis=0)
# data_nom = (data - min) / (max - min)
# data_nom_tensor = torch.tensor(data_nom, dtype=torch.float32)
#
# encoded, decoded = autoencoder(data_nom_tensor)
# autoencoder_MSE = criterion(decoded, data_nom_tensor)
#
# decoded = decoded.detach().numpy()*(max - min) + min
# loc_x = np.loadtxt('loc_x.dat')
# # 生成翼型隐变量数据
# np.savetxt('cpAE_latent80.dat', encoded.detach().numpy(), delimiter='\t')

