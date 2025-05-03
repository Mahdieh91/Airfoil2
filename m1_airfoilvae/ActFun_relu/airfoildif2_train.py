#Changes Made
'''
    Noise Generation: The diffusion_process function now uses a cosine function to create a noise factor.
    Functionality: The noise is multiplied by the generated cosine wave, allowing control over the noise's amplitude over the timesteps.

'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_sizes, latent_dim, activation_function):
        super(DiffusionModel, self).__init__()
        self.activation_function = activation_function

        # Define the model architecture
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_sizes:
            self.layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        self.fc_out = nn.Linear(hidden_sizes[-1], input_dim)

    def forward(self, x, t):
        for layer in self.layers:
            x = layer(x)
            x = self.get_activation(x)
        return self.fc_out(x)

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

def loss_function(recon_x, x):
    return F.mse_loss(recon_x, x)

def diffusion_process(data, timesteps):
    # Create noise using a cosine function
    noise_factor = torch.cos(torch.linspace(0, np.pi, timesteps)).unsqueeze(0)
    noise = torch.randn_like(data) * noise_factor
    return data + noise

def random_search(train_data, input_dim, device, n_iter=20, epochs=1000):
    best_params = None
    best_loss = float('inf')

    for i in range(n_iter):
        hidden_layers = np.random.choice(range(2, 6))
        hidden_sizes = [np.random.choice([50, 100, 150, 200, 250]) for _ in range(hidden_layers)]
        learning_rate = np.random.choice([1e-5, 1e-4, 1e-3, 1e-2])
        activation_function = np.random.choice(['relu'])

        model = DiffusionModel(input_dim=input_dim, hidden_sizes=hidden_sizes, 
                                latent_dim=None, activation_function=activation_function).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Model training
        model.train()
        for epoch in range(epochs):
            timesteps = np.random.randint(1, 100)
            noisy_data = diffusion_process(train_data, timesteps)

            recon = model(noisy_data, timesteps)
            loss = loss_function(recon, train_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Model evaluation
        model.eval()
        with torch.no_grad():
            recon = model(train_data, timesteps)
            relative_loss = loss_function(recon, train_data)

        if relative_loss < best_loss:
            best_loss = relative_loss
            best_params = (hidden_sizes, learning_rate, activation_function)
            torch.save(model.state_dict(), 'best_diffusion_model.pth')

        print(f'Iteration [{i + 1}/{n_iter}], Best params:{best_params}, Best Loss: {best_loss:.4f}')

    return best_params, best_loss

def main_train():
    # Load data and prepare it
    data = np.loadtxt('path_to_your_data.dat')
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_nom = (data - data_min) / (data_max - data_min)
    data_tensor = torch.tensor(data_nom, dtype=torch.float32).to(device)

    # Random search for the best parameters
    best_params, best_loss = random_search(data_tensor, input_dim=data_tensor.shape[1], 
                                           device=device, n_iter=100, epochs=20000)
    print("Best parameters:", best_params)
    print("Best loss:", best_loss)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_train()