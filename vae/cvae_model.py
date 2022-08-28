import torch.nn as nn
import torch.nn.functional as F
import torch

class ConditionalVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 latent_dim,
                 num_classes,
                 image_size,
                 device,
                 hidden_dims = None,
                 features_d = 64):
        super().__init__()
        
        self.device = device
        self.latent_dim = latent_dim
        self.input_channel = in_channels
        self.image_size = image_size
        
        self.embed_class = nn.Embedding(num_classes, self.image_size * self.image_size)
        #self.embed_data = nn.Conv2d(in_channels, in_channels,1)
        
        
            
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels+1, features_d, 4, 2, 1), # 64 -> 32
            nn.InstanceNorm2d(features_d, affine=True),
            nn.LeakyReLU(0.2),
            self._encoder_block(features_d, features_d * 2, 4, 2, 1),# 32 -> 16
            self._encoder_block(features_d * 2, features_d * 4, 4, 2, 1),# 16 - > 8
            self._encoder_block(features_d * 4, features_d * 8, 4, 2, 1), # 8 -> 4
        )

        self.fc_mu = nn.Linear(features_d * 8 * 2 * 2, latent_dim)
        self.fc_var = nn.Linear(features_d * 8 * 2 * 2, latent_dim)
        
        
        self.decoder_input = nn.Linear(latent_dim + 1, features_d * 8 * 2 * 2)
        
        self.decoder = nn.Sequential(
            self._decoder_bloc(features_d * 8, features_d * 4, 4, 2, 1),# 32 -> 16
            self._decoder_bloc(features_d * 4, features_d * 2, 4, 2, 1),# 16 - > 8
            self._decoder_bloc(features_d * 2, features_d, 4, 2, 1), # 8 -> 4
            nn.ConvTranspose2d(features_d, in_channels, 4, 2, 1), # 64 -> 32
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, 1), # 32 -> 64
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        
    def _encoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def _decoder_bloc(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def encode(self,input):
        result = self.encoder(input)
        #print(result.shape)
        result = result.flatten(1)
        
        if torch.isnan(result).any().item():
            print(result)
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return mu, log_var
    
    def decode(self,z):
        result  = self.decoder_input(z)
        #print(result.shape)
        result  = result.view(-1,512,2,2)
        #print(result.shape)
        result = self.decoder(result)
        return result
    
    def reparameterize(self, mu, log_var):
        
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
        
    
    def forward(self,input,label):
        embedded_class = self.embed_class(label)
        embedded_class = embedded_class.view(label.size(0),1,self.image_size,self.image_size)
        #embedded_input = self.embed_data(input) 
        
        x = torch.cat([input,embedded_class],dim=1)
        
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        lab = label.unsqueeze(1)
        
        z = torch.cat([z,lab],dim=1)
        
        return self.decode(z), mu, log_var
    
    def sample(self, n,label):
        label = label.unsqueeze(1)
        z = torch.randn(n, self.latent_dim).to(self.device)
        z = torch.cat([z,label],dim=1)
        #z = z.to(self.device)
        
        return self.decode(z)