import torch
from torch import nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim,device):
        super(CVAE, self).__init__()
        self.device = device
        self.latent_dim = z_dim

        # encoder part
        self.fc1 = nn.Linear(x_dim+1, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim+1, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
    
    def sample(self,n,labels):
        latent = torch.randn(n, self.latent_dim).to(self.device)
        labels = labels.unsqueeze(1).to(self.device)
        
        latent = torch.cat((latent,labels),dim = 1)
        return self.decoder(latent)
        
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
   
    def forward(self, x,labels):
        
        labels = labels.unsqueeze(1)
        x = x.flatten(1)
        x = torch.cat([x,labels],1)
        
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        z = torch.cat([z,labels],1)
        #print(z.shape)
        return self.decoder(z), mu, log_var
    

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim+1, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim+1, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x,labels):
        
        labels = labels.unsqueeze(1)
        x = x.view(-1, 784)
        x = torch.cat([x,labels],1)
        
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        z = torch.cat([z,labels],1)
        #print(z.shape)
        return self.decoder(z), mu, log_var

