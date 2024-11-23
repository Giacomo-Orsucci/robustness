import numpy as np
import os
import PIL
from PIL import Image
import cv2
import torch
from models import StegaStampDecoder
import matplotlib.pyplot as plt
from graphs import plotting
from psnr import main as mainp
from accuracy import main as maina


accuracy_array = []
compression_rate_array = []
psnr_array = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#decoder_path = "/media/giacomo/volume/test_yuv/primo/checkpoints/dec.pth"
decoder_path = "/media/giacomo/volume/yuv_base/enc-dec/checkpoints/dec.pth"


#fingerprint embedded in the images
fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0])

#image_directory = '/media/giacomo/volume/test_yuv/stylegan2_gen_50k_config-e_25'
image_directory = '/media/giacomo/volume/yuv_base/stylegan2_gen_50k_config-e_75_seed42'
img_compressed_path = ""


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


for i in range(100,9,-10):
    j=0
    for filename in os.listdir(image_directory):

        j += 1 #to count the number of images in the folder
        #if j==10: break;
        
        print(j)
        
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            
            img_path = os.path.join(image_directory, filename)
            img = cv2.imread(img_path,3)
            

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_compressed_path = os.path.join("/media/giacomo/volume/yuv_base/robustness_75_seed42/jpeg_compression_quality_100-10_style2_75_50k", f"{i}")
            os.makedirs(img_compressed_path, exist_ok=True)
            png_filename = os.path.join(img_compressed_path, filename)

            base_name, _ = os.path.splitext(png_filename)
            png_filename = base_name + "." + "jpg"

            PIL.Image.fromarray(img, "RGB").save(png_filename,"JPEG", quality=i)

    print(image_directory)
    print(img_compressed_path)
    psnr = mainp(image_directory, img_compressed_path)
    psnr_array.append(psnr)
    compression_rate_array.append(i)
    bitwise_accuracy = maina(img_compressed_path, decoder_path)    
    accuracy_array.append(bitwise_accuracy)
    

print(compression_rate_array)
print(accuracy_array)
print(psnr_array)

plotting(compression_rate_array,accuracy_array,psnr_array,"% of quality","Bitwise accuracy","PSNR (dB)","JPEG compression")

"""
plt.plot(compression_rate_array, accuracy_array, marker='s', linestyle='--', color='black', markerfacecolor='red', markeredgecolor='red')
plt.grid(color='grey', linestyle='-', linewidth=0.5)

plt.yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0]) #to fix the y scale but it can be used also accuracy_array
plt.xticks([100,90,80,70,60,50,40,30,20,10])
plt.gca().invert_xaxis() 


#figure, axis = plt.subplots(1, 1)
#figure.suptitle("Gaussian noise")
#axis.plot(std_array, accuracy_array)
plt.title("JPEG compression", fontweight="bold")
plt.ylabel("Bitwise accuracy")
plt.xlabel("% of quality")
plt.show()
"""
