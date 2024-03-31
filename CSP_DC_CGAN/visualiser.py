import numpy as np
import matplotlib.pyplot as plt

# Example of loading and visualizing the data
data = np.load("./generated_data/epoch_150.npy")  # Load the data from the file
print(data[1,0].shape)
plt.imshow(data[0, 0], aspect='auto')  # Visualize the first sample of the first channel
plt.colorbar()
plt.show()

'''import numpy as np
import mne
import matplotlib.pyplot as plt

# Load the data
data = np.load("./generated_data/epoch_100.npy")

# Check the original shape
print("Original data shape:", data.shape)

# We need to select one set of the data and remove any singleton dimensions
# Let's select the first set (index 0) and squeeze out singleton dimensions
data = np.squeeze(data[0])

# Now the data should be of shape (3, 1000)
# If it's not, we will raise an error
if data.shape != (3, 1000):
    raise ValueError(f"Data after squeezing is not of the expected shape (3, 1000), got {data.shape}")

# Define the info object for MNE
info = mne.create_info(ch_names=['Channel1', 'Channel2', 'Channel3'], sfreq=250, ch_types='eeg')

# Create the RawArray object
raw = mne.io.RawArray(data, info)

# Plot the data using MNE's plotting function
raw.plot_psd()

# Show the figure if not shown automatically
plt.show()
'''