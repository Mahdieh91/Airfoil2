import random
from glob import glob
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

# 各个Latent MSE
Latent_MSE = [3.41289, 1.04498, 0.606603, 0.4414, 0.3760328, 0.439374]
Latent = [2, 5, 10, 20, 40, 80]

plt.figure(figsize=(4, 3.3), dpi=300)
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel('Latent Dimension')
plt.ylabel(r'MSE($\times10^{-5}$)')
plt.xticks(Latent)
plt.plot(Latent, Latent_MSE, linewidth=1.2, color='blue', marker='o', markerfacecolor='white',
         markersize=5, label='Predition')
plt.tight_layout()
output_path = r'E:\D_PHD\D6_Project\pre_cp\AirfoilVAE\image\apd_cvae_latent.png'
plt.savefig(output_path, dpi=300)
plt.show()