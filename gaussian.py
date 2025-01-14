import numpy as np
import os
import PIL
from PIL import Image
import cv2
import torch
from models import StegaStampDecoder
import matplotlib.pyplot as plt
from graphs import plotting
from psnr import main



mean = 0
std = 0

accuracy_array = []
std_array = []
psnr_array = []

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


for i in range(11):
    j=0
    for filename in os.listdir(image_directory):

        j += 1 #to count the number of images in the folder

        print(j)
        if j == 10: break
        
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            
            img_path = os.path.join(image_directory, filename)
            img = cv2.imread(img_path,3)

            x, y, channels = img.shape  # Include the third dimension for color channels

            # Generate noise with the same shape as that of the image
            noise = np.random.normal(loc=mean, scale=std, size=(x, y, channels))  # Adjust noise shape

            # Add the noise to the image
            img_noised = img + noise

            # Clip the pixel values to be between 0 and 255 and convert to uint8
            img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)

            # Convert BGR to RGB
            img_noised_rgb = cv2.cvtColor(img_noised, cv2.COLOR_BGR2RGB)

            img_noised_rgb_array = np.array(img_noised_rgb)
            image_noised_rgb_tensor = torch.from_numpy(img_noised_rgb_array).permute(2, 0, 1).float().to(device)

            detected_fingerprints = RevealNet(image_noised_rgb_tensor.unsqueeze(0))
            detected_fingerprints = (detected_fingerprints > 0).long()
        
            bitwise_accuracy += (detected_fingerprints == fingerprint).float().mean(dim=1).sum().item()
            
            #insert the path where to save the perturbated images
            path_to_save= ""
            img_noise_path = os.path.join(path_to_save, f"{std}") 
            os.makedirs(img_noise_path, exist_ok=True)
            png_filename = os.path.join(img_noise_path, filename)
            PIL.Image.fromarray(img_noised_rgb, "RGB").save(png_filename)
            
            
            img_array = np.array(img)
        
            
    psnr = main(image_directory, img_noise_path)
    psnr_array.append(psnr)
    std_array.append(std)
    std +=10
    bitwise_accuracy = bitwise_accuracy/j
    accuracy_array.append(bitwise_accuracy)

plotting(std_array,accuracy_array,psnr_array,"Noise std","Bitwise accuracy","PSNR (dB)","Gaussian noise")

