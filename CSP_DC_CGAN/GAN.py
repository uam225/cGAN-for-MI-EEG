import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

class Generator(nn.Module):
    def __init__(self, noise_dim=10, feature_dim=3, channels=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + feature_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, channels * 1000),#25 * 342),
            nn.Tanh()
        )

    def forward(self, noise, features):
        x = torch.cat((noise, features), dim=1)
        return self.model(x).view(-1, 1, 3, 1000) #25, 342)


class Discriminator(nn.Module):
    def __init__(self, feature_dim=3, channels=3):
        super(Discriminator, self).__init__()
        image_size_flat = channels * 1000 #channels * 342
        input_dim = image_size_flat + feature_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, features):
        img_flat = img.view(img.size(0), -1)
        x = torch.cat((img_flat, features), dim=1)
        return self.model(x)


class GAN:
    def __init__(self, channels=3, batchsize=20,  noise_dim=100, feature_dim=10):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.channels = channels
        self.batchsize = batchsize
        self.noise_dim = noise_dim  
        self.feature_dim = feature_dim

        self.generator = Generator(noise_dim=noise_dim, feature_dim=feature_dim, channels=channels).to(self.device)
        self.discriminator = Discriminator(channels=channels, feature_dim=feature_dim).to(self.device) #removed channels, may need to add back

        print(f"Number of parameters in discriminator: {len(list(self.discriminator.parameters()))}")

        self.optimiser_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimiser_D = optim.Adam(self.discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))

        self.loss = nn.BCELoss()

    import matplotlib.pyplot as plt

    def train(self, data_loader, epochs, sample_interval=100):
        d_losses, g_losses = [], []
        lr_G = self.optimiser_G.param_groups[0]['lr']  
        lr_D = self.optimiser_D.param_groups[0]['lr']

        for epoch in range(epochs):
            d_loss_sum, g_loss_sum = 0.0, 0.0
            n_batches = 0

            for i, (imgs, features) in enumerate(data_loader):
                valid = torch.ones((imgs.size(0), 1), device=self.device)
                fake = torch.zeros((imgs.size(0), 1), device=self.device)

                real_imgs = imgs.float().to(self.device)
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

            if epoch % sample_interval == 0:
                self.save_samples(epoch, gen_imgs)

        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        os.makedirs('training_plots', exist_ok=True)
        plt.savefig(f'training_plots/training_losses_epochs_{epochs}_LRG_{lr_G}_LRD_{lr_D}.png')
        plt.close()


    '''def train(self, data_loader, epochs):

        d_losses = []
        g_losses = []

        for epoch in range(epochs):
            for i, (imgs, features) in enumerate(data_loader):
                valid = torch.ones((imgs.size(0), 1), device=self.device)
                fake = torch.zeros((imgs.size(0), 1), device=self.device)

                real_imgs = imgs.float().to(self.device)
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
                real_loss = self.loss(self.discriminator(real_imgs, features), valid)
                fake_loss = self.loss(self.discriminator(gen_imgs.detach(), features), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimiser_D.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

                #if i % sample_interval == 0:
                print(f"Epoch: {epoch}, Batch: {i}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

            #if epoch % sample_interval == 0:
                self.save_samples(epoch, gen_imgs)
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        plt.title('Training Losses')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()'''


    def save_samples(self, epoch, gen_imgs):
        os.makedirs('generated_data', exist_ok=True)
        os.makedirs("images", exist_ok=True)
        data = gen_imgs.data.cpu().numpy()
        np.save(f'generated_data/epoch_{epoch}.npy', data)
        img_grid = make_grid(gen_imgs.data[:25], nrow=5, normalize=True)
        plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy(), interpolation='nearest')
        plt.title(f"Epoch {epoch}")
        plt.savefig(f"images/epoch_{epoch}.png")
        plt.close()

