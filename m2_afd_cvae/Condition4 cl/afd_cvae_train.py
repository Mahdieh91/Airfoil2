import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_sizes, latent_dim, activation_function):
        super(CVAE, self).__init__()
        self.activation_function = activation_function

        # Encoder layers
        self.encoders = nn.ModuleList()
        in_dim = input_dim + 1
        for h_dim in hidden_sizes:
            self.encoders.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        self.fc_mu = nn.Linear(hidden_sizes[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_sizes[-1], latent_dim)

        # Decoder layers
        self.decoders = nn.ModuleList()
        in_dim = latent_dim + 1
        for h_dim in reversed(hidden_sizes):
            self.decoders.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        self.fc_out = nn.Linear(hidden_sizes[0], input_dim)

    def encoder(self, x, condition):
        x = torch.cat((x, condition), dim=1)
        for layer in self.encoders:
            x = layer(x)
            x = self.get_activation(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z, condition):
        z = torch.cat((z, condition), dim=1)
        for layer in self.decoders:
            z = layer(z)
            z = self.get_activation(z)
        return self.fc_out(z)

    def forward(self, x, condition):
        mu, logvar = self.encoder(x, condition)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, condition), mu, logvar

    def get_activation(self, x):
        if self.activation_function == 'relu':
            return F.relu(x)
        elif self.activation_function == 'tanh':
            return torch.tanh(x)
        elif self.activation_function == 'leaky_relu':
            return F.leaky_relu(x)
        elif self.activation_function == 'sigmoid':
            return torch.sigmoid(x)
        else:
            raise ValueError("Invalid activation function")


def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return 2*MSE + KLD, MSE, KLD


def random_search(dataset, input_dim, device, n_iter=20, epochs=1000):
    best_params = None
    best_loss = float('inf')

    for i in range(n_iter):
        hidden_layers = 2
        hidden_sizes = [200, 100]
        learning_rate = 0.001
        latent_dim = 4
        activation_function = 'relu'
        # batch_size = [32, 64, 128, 256, 512, 1024, 2048]
        batch_size = [512]
        batch_size = batch_size[i]

        print(f'Iteration [{i + 1}/{n_iter}], Hidden sizes: {hidden_sizes}, Learning rate: {learning_rate}, Latent dim: {latent_dim}, Batch size: {batch_size}, Activation function: {activation_function}')

        cvae = CVAE(input_dim=input_dim, hidden_sizes=hidden_sizes, latent_dim=latent_dim,
                  activation_function=activation_function).to(device)
        optimizer = optim.Adam(cvae.parameters(), lr=learning_rate)

        train_data, test_data = train_test_split(dataset, test_size=0.01, random_state=42)

        batch_size = int(batch_size)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        cvae.train()
        cvae_train_loss = []
        for epoch in range(epochs):

            for train_batch_data in train_loader:
                optimizer.zero_grad()
                recon, mu, logvar = cvae(train_batch_data[:, :199], train_batch_data[:, 199].view(-1, 1))
                loss, mse_loss, kld_loss = loss_function(recon, train_batch_data[:, :199], mu, logvar)
                loss.backward()
                optimizer.step()
                cvae_train_loss.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{epochs}], train_loss: {loss.item():.4f}, '
                    f'MSE Loss: {mse_loss.item():.4f}, KLD Loss: {kld_loss.item():.4f}')

        cvae.eval()
        with torch.no_grad():
            test_data_recon, mu, logvar = cvae(test_data[:, :199], test_data[:, 199].view(-1, 1))
            test_loss, test_mse, test_kld = loss_function(test_data_recon, test_data[:, :199], mu, logvar)

        if test_loss < best_loss:
            best_loss = test_loss
            best_mse = test_mse
            best_kld = test_kld
            best_params = (hidden_sizes, learning_rate, latent_dim, activation_function, batch_size)

            torch.save(cvae.state_dict(), 'best_cvae.pth')
            np.savetxt('cvae_train_loss.dat', cvae_train_loss, delimiter='\t')

            save_params = [str(hidden_sizes), str(learning_rate), str(latent_dim), str(batch_size), str(activation_function), 'best_loss:'+str(best_loss.cpu().detach().item())]
            with open('save_params.txt', 'w') as f:
                f.write('\t'.join(save_params) + '\n')

        print(f'Iteration [{i + 1}/{n_iter}], Best params:{best_params}, Best Loss: {best_loss:.4f}, '
              f'Best mse: {best_mse:.4f}, Best kld: {best_kld:.4f}, Current Loss: {test_loss:.4f}')

    return best_params, best_loss


def main_train():
    data = np.loadtxt('airfoils_recon_data.dat')
    data = data[(data[:, 200] > 0.5) & (data[:, 200] < 1.5)]  # cl
    data = data[(data[:, 201] > 0.01)]  # cd
    data = data[(data[:, 202] > 15) & (data[:, 202] < 65)]  # ld

    airfoil = data[:, :199]

    noise = np.zeros((len(airfoil), 199))
    noise[:, 99] = np.random.normal(0, 1e-7, len(airfoil),)

    airfoil_min = np.min(airfoil, axis=0)
    airfoil_max = np.max(airfoil, axis=0)
    airfoil_nom = (airfoil - airfoil_min) / (airfoil_max - airfoil_min)
    airfoil_nom_tensor = torch.FloatTensor(airfoil_nom).to(device)
    cl = data[:, 200]
    cl_nom = (cl - np.min(cl)) / (np.max(cl) - np.min(cl))
    cl_tensor = torch.FloatTensor(cl_nom).to(device)
    dataset = torch.cat((airfoil_nom_tensor, cl_tensor.view(-1, 1)), dim=1)

    best_params, best_loss = random_search(dataset, input_dim=199, device=device, n_iter=1, epochs=1000)
    print("Best parameters:", best_params)
    print("Best loss:", best_loss)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_train()
