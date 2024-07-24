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

data = np.loadtxt('airfoils_recon_data.dat')
airfoil = data[:, :199]

airfoil_min = np.min(airfoil, axis=0)
airfoil_max = np.max(airfoil, axis=0)
airfoil_nom = (airfoil - airfoil_min) / (airfoil_max - airfoil_min)
airfoil_nom_tensor = torch.FloatTensor(airfoil_nom)

airfoil_flipped = np.fliplr(airfoil)
camber = np.max((airfoil + airfoil_flipped) / 2, axis=1)
camber_tensor = torch.FloatTensor(camber)
dataset = torch.cat((airfoil_nom_tensor, camber_tensor.view(-1, 1)), dim=1)
dataset = dataset[(camber > 0) & (camber < 0.1)]

train_data, test_data = train_test_split(dataset, test_size=0.01, random_state=42)

best_params = ([200, 100], 0.001, 4, 'relu', 512)
hidden_sizes, learning_rate, latent_dim, activation_function, batch_size = best_params

cvae = CVAE(input_dim=199, hidden_sizes=hidden_sizes, latent_dim=latent_dim, activation_function=activation_function)
cvae.load_state_dict(torch.load('best_cvae.pth'))

# generate the selected camber airfoils
cvae.eval()
selected_camber = np.linspace(0, 0.1, 1000)
selected_camber = torch.tensor(selected_camber, dtype=torch.float32)
sample_mu = np.random.normal(loc=0, scale=1.0, size=(len(selected_camber), latent_dim))
sample_mu = torch.tensor(sample_mu, dtype=torch.float32)
recon = cvae.decoder(sample_mu, selected_camber.view(-1, 1))
airfoils_recon = recon.detach().numpy()
airfoils_recon = airfoils_recon * (airfoil_max - airfoil_min) + airfoil_min
airfoils_recon_flipped = np.fliplr(airfoils_recon)
airfoils_recon_camber = np.max((airfoils_recon + airfoils_recon_flipped) / 2, axis=1)

camber_mse = np.mean((airfoils_recon_camber - selected_camber.numpy()) ** 2)
print('camber_mse：', camber_mse)

# r2
y_true = selected_camber.numpy()
y_pred = airfoils_recon_camber
mse = np.mean((y_true - y_pred) ** 2)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print("Coefficient of determination (R^2):", r2)

# MAE
mae = np.mean(np.abs(y_true - y_pred))
print("Mean Absolute Error (MAE):", mae)

cvae_diversity = np.mean(np.var(airfoils_recon, axis=0))
airfoil_filter = np.apply_along_axis(lambda x: savgol_filter(x, 15, 3), axis=1, arr=airfoils_recon)
cvae_roughness = np.mean(np.mean((airfoils_recon - airfoil_filter) ** 2, axis=1))
print(cvae_diversity/0.00038256164690759424)
print(cvae_roughness/2.9165800679528876e-09)
