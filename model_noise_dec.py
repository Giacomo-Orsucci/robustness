import numpy as np
import os
from PIL import Image
import torch
from models import StegaStampDecoder
import matplotlib.pyplot as plt


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

#insert the path of the decoder 
decoder_path = ''


fingerprint = torch.tensor([0,1,0,0,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,1,0,
                            1,1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,0,1,
                            0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,1]).to(device) #embedded fingerprint with seed 42_3

#insert the path of the images to perturbate
image_directory=''



IMAGE_RESOLUTION = 128
IMAGE_CHANNELS = 3
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

    j=0
    
    for filename in os.listdir(image_directory):

        j += 1 #to count the number of images in the folder

        #if j == 10: break;
        
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):

            RevealNet = StegaStampDecoder( #decoder and parameter passing
                IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
            )

            state_dict = torch.load(decoder_path, map_location=device)
            RevealNet.load_state_dict(state_dict)
            RevealNet.to(device)  # Move the model to the device
            RevealNet.eval()      # Set the model to evaluation mode

            param_noise(RevealNet, mean, std)

            img_path = os.path.join(image_directory, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            image_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().to(device)

            detected_fingerprints = RevealNet(image_tensor.unsqueeze(0))
            detected_fingerprints = (detected_fingerprints > 0).long()
        
            bitwise_accuracy += (detected_fingerprints == fingerprint).float().mean(dim=1).sum().item()

            img_array = np.array(img)
            
    std_array.append(std)
    bitwise_accuracy = bitwise_accuracy/j
    accuracy_array.append(bitwise_accuracy)
    

plt.plot(std_array, accuracy_array, marker='s', linestyle='--', color='black', markerfacecolor='red', markeredgecolor='red')
plt.grid(color='grey', linestyle='-', linewidth=0.5)

plt.yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0]) #to fix the y scale but it can be used also accuracy_array
plt.xticks([0,0.05,0.1,0.15,0.2,0.25,0.3])


plt.title("Model noise", fontweight="bold")
plt.ylabel("Bitwise accuracy")
plt.xlabel("Noise std")
plt.show()
