import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from airfoilVAE_train import VAE, loss_function
import pandas as pd
import seaborn as sns

# data load
data = np.loadtxt('airfoils.dat')
data = data[:, 1:]
data = data + noise
data_min = np.min(data, axis=0)
data_max = np.max(data, axis=0)
data_nom = (data - data_min) / (data_max - data_min)
data_tensor = torch.tensor(data_nom, dtype=torch.float32)

# best hyperparameters can be determined from training
best_params = ([200, 250], 0.01, 4, 'relu')
hidden_sizes, learning_rate, latent_dim, activation_function = best_params

vae = VAE(input_dim=199, hidden_sizes=hidden_sizes, latent_dim=latent_dim, activation_function=activation_function)
vae.load_state_dict(torch.load('best_vae.pth'))
vae.eval()

# randomly generate 1000 airfoils
airfoils_recon = []
for j in range(1000):
    sample_mu = np.random.normal(loc=0, scale=1.2, size=(len(data), latent_dim))
    sample_mu = torch.tensor(sample_mu, dtype=torch.float32)
    recon = vae.decoder(sample_mu)
    if len(airfoils_recon) == 0:
        airfoils_recon = recon.detach().numpy()
    else:
        airfoils_recon = np.vstack((airfoils_recon, recon.detach().numpy()))
airfoils_recon = airfoils_recon * (data_max - data_min) + data_min

# compute the relative diversity and relative roughness
VAE_diversity = np.mean(np.var(airfoils_recon, axis=0))
airfoil_filter = np.apply_along_axis(lambda x: savgol_filter(x, 15, 3), axis=1, arr=airfoils_recon)
VAE_roughness = np.mean(np.mean((airfoils_recon - airfoil_filter) ** 2, axis=1))
print(VAE_diversity/0.00038256164690759424)
print(VAE_roughness/2.9165800679528876e-09)
