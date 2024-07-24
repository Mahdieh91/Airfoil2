import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

# define VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_sizes, latent_dim, activation_function):
        super(VAE, self).__init__()
        self.activation_function = activation_function

        # Encoder layers
        self.encoders = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_sizes:
            self.encoders.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        self.fc_mu = nn.Linear(hidden_sizes[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_sizes[-1], latent_dim)

        # Decoder layers
        self.decoders = nn.ModuleList()
        in_dim = latent_dim
        for h_dim in reversed(hidden_sizes):
            self.decoders.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        self.fc_out = nn.Linear(hidden_sizes[0], input_dim)

    def encoder(self, x):
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

    def decoder(self, z):
        for layer in self.decoders:
            z = layer(z)
            z = self.get_activation(z)
        return self.fc_out(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

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


def compute_relative_metrics(recon_x, data_min, data_max, noise):
    recon_np = recon_x.cpu().detach().numpy()

    airfoils_recon = recon_np * (data_max - data_min) + data_min
    airfoils_recon = airfoils_recon - noise

    relative_diversity = np.mean(np.var(airfoils_recon, axis=0)) / 0.00038256164690759424

    airfoils_recon_filter = np.apply_along_axis(lambda x: savgol_filter(x, 15, 3), axis=1, arr=airfoils_recon)
    relative_roughness = np.mean(np.mean((airfoils_recon - airfoils_recon_filter) ** 2, axis=1) / 2.9165800679528876e-09)

    relative_roughness = torch.tensor(relative_roughness, dtype=torch.float32, device=recon_x.device)
    relative_diversity = torch.tensor(relative_diversity, dtype=torch.float32, device=recon_x.device)

    return relative_diversity, relative_roughness


def random_search(train_data, input_dim, device, n_iter=20, epochs=1000, data_min=1e5, data_max=1e5, noise=1e5):
    best_params = None
    best_loss = float('inf')

    for i in range(n_iter):
        # hyperparameters search space
        hidden_layers = np.random.choice(range(2, 6))
        hidden_sizes = [np.random.choice([50, 100, 150, 200, 250]) for _ in range(hidden_layers)]
        learning_rate = np.random.choice([1e-5, 1e-4, 1e-3, 1e-2])
        latent_dim = np.random.choice([2, 4, 8, 16, 32])
        # activation_function = np.random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu'])
        activation_function = np.random.choice(['relu'])

        vae = VAE(input_dim=input_dim, hidden_sizes=hidden_sizes, latent_dim=latent_dim,
                  activation_function=activation_function).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

        # model train
        vae.train()
        VAE_train_loss = []
        for epoch in range(epochs):
            recon, mu, logvar = vae(train_data)
            loss, mse_loss, kld_loss = loss_function(recon, train_data, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            VAE_train_loss.append(loss.item())

            # if (epoch + 1) % 100 == 0:
            #     print(
            #         f'Epoch [{epoch + 1}/{epochs}], Total Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, KLD Loss: {kld_loss.item():.4f}')

        # model test
        vae.eval()
        with torch.no_grad():
            recon, mu, logvar = vae(train_data)
            relative_diversity, relative_roughness = compute_relative_metrics(recon, data_min, data_max, noise)
            relative_loss = torch.sum((relative_roughness - 1) ** 2 + (relative_diversity - 1) ** 2)


        # save the best model
        if relative_loss < best_loss:
            best_loss = relative_loss
            best_params = (hidden_sizes, learning_rate, latent_dim, activation_function)

            torch.save(vae.state_dict(), 'best_vae.pth')
            np.savetxt('VAE_train_loss.dat', VAE_train_loss, delimiter='\t')

            save_params = [str(hidden_sizes), str(learning_rate), str(latent_dim), str(activation_function), 'best_loss:'+str(best_loss.cpu().detach().item())]
            with open('save_params.txt', 'w') as f:
                f.write('\t'.join(save_params) + '\n')

        print(f'Iteration [{i + 1}/{n_iter}], Best params:{best_params}, Best Loss: {best_loss:.4f}, Current Loss: {relative_loss:.4f}')

    return best_params, best_loss


def main_train():
    # add noize to the zero coordinate
    noise = np.zeros((1500, 199))
    noise[:, 99] = np.random.normal(0, 1e-5, 1500)
    # data load
    data = np.loadtxt('airfoils.dat')
    data = data[:, 1:]
    data = data + noise
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_nom = (data - data_min) / (data_max - data_min)
    data_tensor = torch.tensor(data_nom, dtype=torch.float32).to(device)

    # randomly search
    best_params, best_loss = random_search(data_tensor, input_dim=199, device=device, n_iter=100, epochs=20000,
                                           data_min=data_min, data_max=data_max, noise=noise)
    print("Best parameters:", best_params)
    print("Best loss:", best_loss)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_train()
