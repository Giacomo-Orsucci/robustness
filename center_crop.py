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

accuracy_array = []
crop_size_array = []
psnr_array = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

decoder_path = "/home/giacomo/Desktop/enc_dec_pretrained_celeba/dec.pth"

#fingerprint embedded in the images
fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0])

image_directory = '/media/giacomo/hdd_ubuntu/stylegan2_gen_50k'


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


for i in range(128,10,-8):
    j=0
    for filename in os.listdir(image_directory):

        j += 1 #to count the number of images in the folder

        #if j == 11: break
        
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):

            print("Dimensione")
            print(i)
            
            img_path = os.path.join(image_directory, filename)
            img = Image.open(img_path)

            width, height = img.size

            new_width = i
            new_height = i

            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

            # Crop the center of the image
            img_cropped = img.crop((left, top, right, bottom))

            final_width, final_height = img_cropped.size
            left_pad = (128 - final_width) // 2
            top_pad = (128 - final_height) // 2
    
            # Create a new image with the final size and a black background
            img_final = Image.new("RGB", (128, 128), (0, 0, 0))

            # Paste the resized image onto the black background
            img_final.paste(img_cropped, (left_pad, top_pad))


            img_crop_path = os.path.join("/media/giacomo/hdd_ubuntu/jpeg_cropsize_128-63_style2_50k", f"{i}") 
            os.makedirs(img_crop_path, exist_ok=True)
            img_filename = os.path.join(img_crop_path, filename)
            img_final.save(img_filename)

            img_path = os.path.join(img_crop_path, filename)
            img = Image.open(img_path)

            img_array = np.array(img)
            image_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().to(device)

            detected_fingerprints = RevealNet(image_tensor.unsqueeze(0))
            detected_fingerprints = (detected_fingerprints > 0).long()
        
            print(detected_fingerprints)
            bitwise_accuracy += (detected_fingerprints == fingerprint).float().mean(dim=1).sum().item()

            img_array = np.array(img)
        
            print("img_array")
            print(img_array)
            
    psnr = main(image_directory, img_crop_path)
    psnr_array.append(psnr)
    crop_size_array.append(i)
    bitwise_accuracy = bitwise_accuracy/j
    accuracy_array.append(bitwise_accuracy)
    

print(crop_size_array)
print(accuracy_array)
print(psnr_array)
"""
plt.plot(crop_size_array, accuracy_array, marker='s', linestyle='--', color='black', markerfacecolor='red', markeredgecolor='red')
plt.grid(color='grey', linestyle='-', linewidth=0.5)

plt.yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0]) #to fix the y scale but it can be used also accuracy_array
plt.gca().invert_xaxis() 

plt.title("Center cropping", fontweight="bold")
plt.ylabel("Bitwise accuracy")
plt.xlabel("Crop size")
plt.show()
"""

plotting(crop_size_array,accuracy_array,psnr_array,"Crop size","Bitwise accuracy","PSNR (dB)","Center cropping")