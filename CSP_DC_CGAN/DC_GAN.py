import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


class Generator(nn.Module):
    def __init__(self, noise_dim=100, feature_dim=3, channels=3):
        super(Generator, self).__init__()
        self.init_size = 8  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(noise_dim + feature_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=4),
            nn.Conv1d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, features):
        x = torch.cat((noise, features), dim=1)
        x = self.l1(x)
        x = x.view(x.shape[0], 128, self.init_size**2)#, self.init_size)
        img = self.conv_blocks(x)
        print("Generator image shape:", img.shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, feature_dim=3, channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channels+3, 64, 3, stride=2, padding=1),  # Adding 1 channel for features
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 * 16, 1),  # Adjust size 
            nn.Sigmoid()
        )

    def forward(self, img, features):
        features = features.unsqueeze(2)#.unsqueeze(3)
        #print("Image shape:", img.shape)
        #print("Features shape before expand:", features.shape)
        features = features.expand(img.size(0), features.size(1), img.size(2))#, img.size(3))
        x = torch.cat((img, features), 1)
        return self.model(x)
    
class DCGAN:
    def __init__(self, channels=3, batchsize=50, noise_dim=100, feature_dim=10):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.channels = channels
        self.batchsize = batchsize
        self.noise_dim = noise_dim  
        self.feature_dim = feature_dim

        self.generator = Generator(noise_dim=noise_dim, feature_dim=feature_dim, channels=channels).to(self.device)
        self.discriminator = Discriminator(feature_dim=feature_dim).to(self.device) 

        #print(f"Number of parameters in discriminator: {len(list(self.discriminator.parameters()))}")

        self.optimiser_G = optim.Adam(self.generator.parameters(), lr=0.002, betas=(0.5, 0.999)) #defaults: lr = 0.0002, betas=(0.5, 0.999)
        self.optimiser_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)) #defaults: lr = 0.0002, betas=(0.5, 0.999)

        self.loss = nn.BCELoss()

    def train(self, data_loader, epochs): #removed ", sample_interval=100" as arg
        d_losses, g_losses = [], []
        lr_G = self.optimiser_G.param_groups[0]['lr']  
        lr_D = self.optimiser_D.param_groups[0]['lr']

        for epoch in range(epochs):
            d_loss_sum, g_loss_sum = 0.0, 0.0
            n_batches = 0

            for i, (imgs, features) in enumerate(data_loader):
                imgs_unflattened = imgs.view(-1, 3, 1000)
                valid = torch.ones((imgs_unflattened.size(0), 1), device=self.device)
                fake = torch.zeros((imgs_unflattened.size(0), 1), device=self.device)

                #print('Image size:', imgs_unflattened.shape)
                real_imgs = imgs_unflattened.float().to(self.device)
                features = features.float().to(self.device)

                # Generator
                self.optimiser_G.zero_grad()
                noise = torch.randn(imgs.size(0), self.noise_dim, device=self.device)
                gen_imgs = self.generator(noise, features)
                g_loss = self.loss(self.discriminator(gen_imgs, features), valid)
                g_loss.backward()
                self.optimiser_G.step()

                # Discriminator
                self.optimiser_D.zero_grad()
                #print('Real image size:', real_imgs.shape)
                real_loss = self.loss(self.discriminator(real_imgs, features), valid)
                fake_loss = self.loss(self.discriminator(gen_imgs.detach(), features), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimiser_D.step()

                d_loss_sum += d_loss.item()
                g_loss_sum += g_loss.item()
                n_batches += 1

                #if i % sample_interval == 0:
                print(f"Epoch: {epoch}, Batch: {i}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

            d_losses.append(d_loss_sum / n_batches)
            g_losses.append(g_loss_sum / n_batches)

            #if epoch % sample_interval == 0:
             #   self.save_samples(epoch, gen_imgs)

        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        os.makedirs('training_plots', exist_ok=True)
        plt.savefig(f'training_plots/training_losses_epochs_{epochs}_LRG_{lr_G}_LRD_{lr_D}.png')
        plt.show()
        plt.close()

'''class Generator(nn.Module):
    def __init__(self, noise_dim=100, feature_dim=3, time_points=1000, channels=3):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim + feature_dim, 6200)  # example dimension, adjust based on architecture
        self.fc2 = nn.Linear(6200, 100*62)  # Adjusted for reshaping
        self.batch_norm = nn.BatchNorm1d(100)
        self.conv1 = nn.Conv1d(100, 62, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(62, channels, kernel_size=5, padding=2)
        self.upsample = nn.Upsample(scale_factor=10, mode='nearest')  # To match time points

    def forward(self, noise, features):
        x = torch.cat((noise, features), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.batch_norm(x.view(-1, 100, 62))  # Reshape to (batch, channels, length)
        x = torch.relu(self.conv1(x))
        x = self.upsample(torch.relu(self.conv2(x)))  # Upsample to match EEG data shape
        return x'''
'''class Discriminator(nn.Module):
    def __init__(self, feature_dim=3, time_points=1000, channels=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(channels + feature_dim, 62, kernel_size=5, padding=2)  # Account for features
        self.conv2 = nn.Conv1d(62, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2, stride=2)
        #self.fc1 = nn.Linear(128 * (time_points // 4), 400)  # Adjust based on pooling and input dimensions
        self.fc1 = nn.Linear(128 * 155, 400)  # Adjust based on pooling and input dimensions
        self.fc2 = nn.Linear(400, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, features):
        print(x.shape)
        features = features.unsqueeze(-1)
        features = features.expand(-1, -1, x.size(2))
        print(features.shape)
        x = torch.cat((x, features), dim=1)  # Combine x and features before processing
        x = self.tanh(self.pool(self.conv1(x)))
        #print(x.shape)
        x = self.tanh(self.pool(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.tanh(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x'''