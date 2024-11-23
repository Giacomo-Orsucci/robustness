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

#To do: pulire il codice dato che tante cose sono inutili visto che l'accuratezza si calcola sulle immagini nella cartella

accuracy_array = []
size_array = []
psnr_array = []


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

decoder_path = "/media/giacomo/volume/yuv_base/enc-dec/checkpoints/dec.pth"

#fingerprint embedded in the images
fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0])

image_directory = '/media/giacomo/volume/yuv_base/stylegan2_gen_50k_config-e_75_seed42'


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
#for i in range(1,75,8):
k=0
for i in range(1,9):

    k = i;

    if i % 2 == 0: #to ensure that the kernel has odd dimensions. It is mandatory to use the following blurring function
        k+=1
        
    j=0
    for filename in os.listdir(image_directory):

        j += 1 #to count the number of images in the folder
        print(j)


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

            print("img-noise-yuv")
            print(img_blurred_yuv.shape)

            img_blurred_rgb_array = np.array(img_blurred_rgb) #to convert in array
            image_blurred_rgb_tensor = torch.from_numpy(img_blurred_rgb_array).permute(2, 0, 1).float().to(device) #to convert in tensor

            image_blurred_yuv_tensor = torch.from_numpy(img_blurred_yuv).float().unsqueeze(0)
            print("dimensione tensore per firma")
            print(image_blurred_yuv_tensor.shape)
            y_channel_list = []
            y_channel_list.append(image_blurred_yuv_tensor)

           
            images_y_batch = []
            images_y_batch = torch.stack(y_channel_list).to(device)

            print("batch shape")
            print(images_y_batch.shape)

            detected_fingerprints = RevealNet(images_y_batch)
            detected_fingerprints = (detected_fingerprints > 0).long()
        
            #print(detected_fingerprints)
            

            #img_blurred_path = os.path.join("/media/giacomo/volume/test_yuv/robustness/gau_blurring_size_1-73_style2_25_50k", f"{k}")
            img_blurred_path = os.path.join("/media/giacomo/volume/yuv_base/robustness_75_seed42/gau_blurring_size_1-9_style2_75_50k", f"{k}")
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


