import numpy as np
import os
from PIL import Image
import torch
from models import StegaStampDecoder
import matplotlib.pyplot as plt
from accuracy import main as maina


#Function to add gaussian noise to decoder's parameters
def param_noise(model, mean, std):
    noise = np.random.normal(loc=mean, scale=std)  

    for param in model.parameters():
        if param.requires_grad:
            param.data += noise


mean = 0
std = 0

accuracy_array = []
std_array = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#insert the path of the decoder that you want to use
decoder_path = ""


#fingerprint embedded in the images
fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0])


#insert the path of the images that you want to perturbate
image_directory = ''



IMAGE_RESOLUTION = 128
IMAGE_CHANNELS = 1
FINGERPRINT_SIZE = len(fingerprint)

RevealNet = StegaStampDecoder( #decoder and parameter passing
    IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
)

state_dict = torch.load(decoder_path, map_location=device)
RevealNet.load_state_dict(state_dict)
RevealNet.to(device)  # Move the model to the device
RevealNet.eval()      # Set the model to evaluation mode

bitwise_accuracy = 0
fingerprint = (fingerprint > 0).long().to(device)

for i in range(0,31,5):

    std = i/100

    RevealNet = StegaStampDecoder( #decoder and parameter passing
                IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
            )

    state_dict = torch.load(decoder_path, map_location=device)
    RevealNet.load_state_dict(state_dict)
    RevealNet.to(device)  # Move the model to the device
    RevealNet.eval()      # Set the model to evaluation mode
    param_noise(RevealNet, mean, std)
    std_array.append(std)
    bitwise_accuracy = maina(image_directory, decoder_path, RevealNet)
    accuracy_array.append(bitwise_accuracy)
    


plt.plot(std_array, accuracy_array, marker='s', linestyle='--', color='black', markerfacecolor='red', markeredgecolor='red')
plt.grid(color='grey', linestyle='-', linewidth=0.5)

plt.yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0]) #to fix the y scale but it can be used also accuracy_array
plt.xticks([0,0.05,0.1,0.15,0.2,0.25,0.3])


plt.title("Model noise", fontweight="bold")
plt.ylabel("Bitwise accuracy")
plt.xlabel("Noise std")
plt.show()
