import numpy as np
import os
import PIL
from PIL import Image
import cv2
import torch
from models import StegaStampDecoder
import matplotlib.pyplot as plt
from psnr import main as mainp
from accuracy import main as maina
from graphs import plotting


accuracy_array = []
size_array = []
psnr_array = []


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#insert the path of the decoder that you want to use
decoder_path = ""

#fingerprint embedded in the images
fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0]) #fin embedded with seed=42
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

img_blurred_path=" "
#for i in range(1,9):
k=0
for i in range(1,75,8):

    k = i;

    #to ensure that the kernel has odd dimensions. It is mandatory to use the following blurring function
    #if i % 2 == 0: k+=1
        
    j=0
    for filename in os.listdir(image_directory):

        j += 1 #to count the number of images in the folder
        


        #if j == 10: break #to ensure a little generation to try the code
        
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            
            img_path = os.path.join(image_directory, filename)
            img = cv2.imread(img_path,3)
            
            # Generates the blurred image applying gaussian blurring using a kernel of size kxk
            blur = cv2.GaussianBlur(img,(k,k),0)

            # Convert BGR to RGB
            img_blurred_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
            img_blurred_yuv = cv2.cvtColor(blur, cv2.COLOR_BGR2YUV)

            y_channel, u_channel, v_channel = cv2.split(img_blurred_yuv)
            img_blurred_yuv = y_channel


            img_blurred_rgb_array = np.array(img_blurred_rgb) #to convert in array
            image_blurred_rgb_tensor = torch.from_numpy(img_blurred_rgb_array).permute(2, 0, 1).float().to(device) #to convert in tensor

            image_blurred_yuv_tensor = torch.from_numpy(img_blurred_yuv).float().unsqueeze(0)
            
            y_channel_list = []
            y_channel_list.append(image_blurred_yuv_tensor)

           
            images_y_batch = []
            images_y_batch = torch.stack(y_channel_list).to(device)


            detected_fingerprints = RevealNet(images_y_batch)
            detected_fingerprints = (detected_fingerprints > 0).long()
            
            #path where you want to save all the perturbated images
            path_to_save = ""
            img_blurred_path = os.path.join(path_to_save, f"{k}")
            os.makedirs(img_blurred_path , exist_ok=True)
            png_filename = os.path.join(img_blurred_path, filename)
            PIL.Image.fromarray(img_blurred_rgb_array, "RGB").save(png_filename)
            l=k
            
            
    psnr = mainp(image_directory, img_blurred_path)
    psnr_array.append(psnr)
    size_array.append(k)
    bitwise_accuracy = maina(img_blurred_path, decoder_path)
    accuracy_array.append(bitwise_accuracy)
    

print(size_array)
print(accuracy_array)
print(psnr_array)

plotting(size_array,accuracy_array,psnr_array,"Kernel size","Bitwise accuracy","PSNR (dB)","Gaussian blurring")


