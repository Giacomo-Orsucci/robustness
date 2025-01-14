import numpy as np
import os
import PIL
from PIL import Image
import cv2
import torch
from models import StegaStampDecoder
import matplotlib.pyplot as plt
from graphs import plotting_center_jpeg
from psnr import main as mainp
from accuracy import main as maina

accuracy_array = []
crop_size_array = []
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


bitwise_accuracy = 0


for i in range(128,10,-8):
    j=0
    for filename in os.listdir(image_directory):

        j += 1 #to count the number of images in the folder

        #to ensure that the kernel has odd dimensions. It is mandatory to use the following blurring function
        #if j == 10: break 
        
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):

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


            #path where you want to save all the perturbated images
            path_to_save = ""
            img_crop_path = os.path.join(path_to_save, f"{i}") 
            os.makedirs(img_crop_path, exist_ok=True)
            img_filename = os.path.join(img_crop_path, filename)
            img_final.save(img_filename)

            

    psnr = mainp(image_directory, img_crop_path)
    psnr_array.append(psnr)
    crop_size_array.append(i)
    bitwise_accuracy = maina(img_crop_path, decoder_path)
    accuracy_array.append(bitwise_accuracy)
    

plotting_center_jpeg(crop_size_array,accuracy_array,psnr_array,"Crop size","Bitwise accuracy","PSNR (dB)","Center cropping")