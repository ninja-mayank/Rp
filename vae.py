import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.distributions import StudentT

# Define the VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        # Degrees of freedom for Student's t-distribution, initialized to 4
        self.v = nn.Parameter(torch.tensor(4.0), requires_grad=True)
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # Use Student's t-distribution with learnable degrees of freedom
        t_dist = StudentT(df=self.v, loc=mu, scale=std)
        return t_dist.rsample()
    
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training function
def train(model, dataloader, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            data = data[0]  # Access the actual data tensor if it's wrapped in a tuple
            data = data.view(-1, 1)  # Ensure data is 2D (batch_size, input_dim)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        # print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')

# Load your tabular data (assuming it's in a CSV file)
def synthetic_data(df):
    data = df.values.astype(np.float32)
    # Normalize data to [0, 1]
    data_min = data.min()
    data_max = data.max()
    data = (data - data_min) / (data_max - data_min)

    # Convert to PyTorch tensor
    data_tensor = torch.tensor(data).reshape(-1, 1)
    # Create DataLoader
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(data_tensor, batch_size=batch_size, shuffle=True)

    # Model parameters
    input_dim = 1
    hidden_dim = 128
    latent_dim = 2

    # Instantiate the model, optimizer
    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    train(model, dataloader, optimizer, num_epochs=50)

    # Generate synthetic data
    model.eval()
    with torch.no_grad():
        z = StudentT(df=model.v.item()).rsample((497, latent_dim))  # Sample from the Student's t-distribution
        generated_data = model.decode(z).numpy()

    # Denormalize generated data back to original range
    generated_data = generated_data * (data_max - data_min) + data_min
    generated_data = generated_data.tolist()

    output_array = [item[0] for item in generated_data]

    return output_array
