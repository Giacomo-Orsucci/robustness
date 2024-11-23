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

#To do: pulire il codice dato che tante cose sono inutili visto che l'accuratezza si calcola sulle immagini nella cartella

mean = 0
std = 0

accuracy_array = []
std_array = []
psnr_array = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#decoder_path = "/home/giacomo/Desktop/enc_dec_pretrained_celeba/dec.pth"
decoder_path = "/media/giacomo/volume/yuv_base/enc-dec/checkpoints/dec.pth"

#fingerprint embedded in the images
fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0])

#image_directory = '/media/giacomo/hdd_ubuntu/stylegan2_gen_50k'
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


for i in range(11):
    #if i ==1:break;
    j=0
    for filename in os.listdir(image_directory):
        y_channel_list = []

        j += 1 #to count the number of images in the folder

        print(j)
        #if j == 100: break
        
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            
            img_path = os.path.join(image_directory, filename)
            img = cv2.imread(img_path,3)
            
            #img = img/255 #if we want the images in greyscale

            x, y, channels = img.shape  # Include the third dimension for color channels

            # Generate noise with the same shape as that of the image
            noise = np.random.normal(loc=mean, scale=std, size=(x, y, channels))  # Adjust noise shape

            # Add the noise to the image
            img_noised = img + noise

            # Clip the pixel values to be between 0 and 255 and convert to uint8
            img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)

            # Convert BGR to RGB
            img_noised_rgb = cv2.cvtColor(img_noised, cv2.COLOR_BGR2RGB)
             # Convert BGR to YUV
            img_noised_yuv = cv2.cvtColor(img_noised, cv2.COLOR_BGR2YUV)
            y_channel, u_channel, v_channel = cv2.split(img_noised_yuv)
            img_noised_yuv = y_channel

            print("img-noise-yuv")
            print(img_noised_yuv.shape)

            img_noised_rgb_array = np.array(img_noised_rgb)
            image_noised_rgb_tensor = torch.from_numpy(img_noised_rgb_array).permute(2, 0, 1).float().to(device)

            
            image_noised_yuv_tensor = torch.from_numpy(img_noised_yuv).float().unsqueeze(0)
            print("dimensione tensore per firma")
            print(image_noised_yuv_tensor.shape)
            y_channel_list = []
            y_channel_list.append(image_noised_yuv_tensor)

           
            images_y_batch = torch.stack(y_channel_list).to(device)

            print("batch shape")
            print(images_y_batch.shape)

            detected_fingerprints = RevealNet(images_y_batch)
            detected_fingerprints = (detected_fingerprints > 0).long()

            print("Fingerprint_shape")
            print(detected_fingerprints.shape)
        
            print(detected_fingerprints)
            
            img_noise_path = os.path.join("/media/giacomo/volume/yuv_base/robustness_75_seed42/gau_noise_std_0-100_style2_75_50k", f"{std}") 
            os.makedirs(img_noise_path, exist_ok=True)
            png_filename = os.path.join(img_noise_path, filename)
            PIL.Image.fromarray(img_noised_rgb, "RGB").save(png_filename)
           
            
    psnr = mainp(image_directory, img_noise_path)
    psnr_array.append(psnr)
    std_array.append(std)
    bitwise_accuracy = maina(img_noise_path, decoder_path)
    accuracy_array.append(bitwise_accuracy)
    std +=10
    

print(std_array)
print(accuracy_array)
print(psnr_array)

plotting(std_array,accuracy_array,psnr_array,"Noise std","Bitwise accuracy","PSNR (dB)","Gaussian noise")