import numpy as np
import os
import PIL
from PIL import Image
import cv2
import torch
from models import StegaStampDecoder
import matplotlib.pyplot as plt
from graphs import plotting
from psnr import main_j as mainp_j
from accuracy import main as maina


accuracy_array = []
compression_rate_array = []
psnr_array = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#insert the path of the decoder that you want to use
decoder_path = ""


#fingerprint embedded in the images
fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0])

#insert the path of the images that you want to perturbate
image_directory = ''




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
        
        
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            
            img_path = os.path.join(image_directory, filename)
            img = cv2.imread(img_path,3)
            

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #path where you want to save all the perturbated images
            path_to_save = ""
            img_compressed_path = os.path.join(path_to_save, f"{i}")
            os.makedirs(img_compressed_path, exist_ok=True)
            png_filename = os.path.join(img_compressed_path, filename)

            base_name, _ = os.path.splitext(png_filename)
            png_filename = base_name + "." + "jpg"

            PIL.Image.fromarray(img, "RGB").save(png_filename,"JPEG", quality=i)

    psnr = mainp_j(image_directory, img_compressed_path)
    psnr_array.append(psnr)
    compression_rate_array.append(i)
    bitwise_accuracy = maina(img_compressed_path, decoder_path)    
    accuracy_array.append(bitwise_accuracy)
    

plotting(compression_rate_array,accuracy_array,psnr_array,"% of quality","Bitwise accuracy","PSNR (dB)","JPEG compression")

