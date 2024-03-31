from data_loader import EEGFeatureDataset
from DC_GAN import DCGAN
#from GAN import GAN
from torch.utils.data import DataLoader

# Set the data and feature directories
data_dir = '/Users/umairarshad/SHU/or_data/mat'
feature_dir = '/Users/umairarshad/SHU/or_data/features'  

# Initialize dataset
subjects = range(6,8)  # Assuming 25 subjects
sessions = range(1, 6)  # Assuming 5 sessions each
dataset = EEGFeatureDataset(data_dir, feature_dir, subjects, sessions)
print(dataset.__len__())

# Initialize data loader
data_loader = DataLoader(dataset, batch_size=20, shuffle=True)
print(f"Total number of batches: {len(data_loader)}")


# Initialize GAN
dcgan = DCGAN(batchsize=10, noise_dim=100, feature_dim=3)  
#gan = GAN(noise_dim=100, feature_dim=3)

# Train GAN
epochs = 200 
dcgan.train(data_loader, epochs)
#gan.train(data_loader, epochs)
