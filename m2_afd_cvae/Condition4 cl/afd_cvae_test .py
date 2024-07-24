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

data = np.loadtxt('airfoils_recon_data.dat')
data = data[(data[:, 200] > 0.5) & (data[:, 200] < 1.5)]  # cl
data = data[(data[:, 201] > 0.01)]  # cd
data = data[(data[:, 202] > 15) & (data[:, 202] < 65)]  # ld
loc_x = np.loadtxt('loc_x.dat')
airfoil = data[:, :199]

airfoil_min = np.min(airfoil, axis=0)
airfoil_max = np.max(airfoil, axis=0)
airfoil_nom = (airfoil - airfoil_min) / (airfoil_max - airfoil_min)
airfoil_nom_tensor = torch.FloatTensor(airfoil_nom)
cl = data[:, 200]
cl_nom = (cl - np.min(cl)) / (np.max(cl) - np.min(cl))
cl_tensor = torch.FloatTensor(cl_nom)
dataset = torch.cat((airfoil_nom_tensor, cl_tensor.view(-1, 1)), dim=1)

train_data, test_data = train_test_split(dataset, test_size=0.01, random_state=42)

best_params = ([200, 100], 0.001, 4, 'relu', 512)
hidden_sizes, learning_rate, latent_dim, activation_function, batch_size = best_params

cvae = CVAE(input_dim=199, hidden_sizes=hidden_sizes, latent_dim=latent_dim, activation_function=activation_function)
cvae.load_state_dict(torch.load('best_cvae.pth'))

# generate the selected cl airfoils
cvae.eval()
selected_cl = np.linspace(0, 1, 1000)
selected_cl = torch.tensor(selected_cl, dtype=torch.float32)
sample_mu = np.random.normal(loc=0, scale=1.0, size=(len(selected_cl), latent_dim))
sample_mu = torch.tensor(sample_mu, dtype=torch.float32)
recon = cvae.decoder(sample_mu, selected_cl.view(-1, 1))
airfoils_recon = recon.detach().numpy()
airfoils_recon = airfoils_recon * (airfoil_max - airfoil_min) + airfoil_min

cvae_diversity = np.mean(np.var(airfoils_recon, axis=0))
airfoil_filter = np.apply_along_axis(lambda x: savgol_filter(x, 15, 3), axis=1, arr=airfoils_recon)
cvae_roughness = np.mean(np.mean((airfoils_recon - airfoil_filter) ** 2, axis=1))
print(cvae_diversity/0.00038256164690759424)
print(cvae_roughness/2.9165800679528876e-09)

selected_cl = np.linspace(0, 1, 1000)*(np.max(cl) - np.min(cl))+np.min(cl)
np.savetxt('airfoils_cl_recon.dat', airfoils_recon, fmt='%0.6f')
np.savetxt('selected_cl.dat', selected_cl, fmt='%0.6f')
time.sleep(1)
subprocess.run(["python", "1getData.py"])
time.sleep(4)

cl_recon_data = np.loadtxt('airfoils_cl_recon_data.dat')
cl_recon_data_airfoil = cl_recon_data[:, :199]
selected_cl = cl_recon_data[:, 200]
rencon_cl = cl_recon_data[:, 201]

cl_mse = np.mean((selected_cl - rencon_cl) ** 2)
print('cl MSE：', cl_mse)

# r2
y_true = selected_cl
y_pred = rencon_cl
mse = np.mean((y_true - y_pred) ** 2)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print("Coefficient of determination (R^2):", r2)

# MAE
mae = np.mean(np.abs(y_true - y_pred))
print("Mean Absolute Error (MAE):", mae)
