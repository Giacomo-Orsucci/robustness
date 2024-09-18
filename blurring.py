import numpy as np
import os
import PIL
from PIL import Image
import cv2
import torch
from models import StegaStampDecoder
import matplotlib.pyplot as plt
from psnr import main
from graphs import plotting



accuracy_array = []
size_array = []
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

img_noise_path=" "
for i in range(1,75,8):

    k = i;

    if i % 2 == 0: #to ensure that the kernel has odd dimensions. It is mandatory to use the following blurring function
        k+=1
        
    j=0
    for filename in os.listdir(image_directory):

        j += 1 #to count the number of images in the folder
        print(j)


        #if j == 11: break #to ensure a little generation to try the code
        
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            
            img_path = os.path.join(image_directory, filename)
            img = cv2.imread(img_path,3)
            
            # Generates the blurred image applying gaussian blurring using a kernel of size kxk
            blur = cv2.GaussianBlur(img,(k,k),0)

            # Convert BGR to RGB
            img_blurred_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

            img_blurred_rgb_array = np.array(img_blurred_rgb) #to convert in array
            image_blurred_rgb_tensor = torch.from_numpy(img_blurred_rgb_array).permute(2, 0, 1).float().to(device) #to convert in tensor

            detected_fingerprints = RevealNet(image_blurred_rgb_tensor.unsqueeze(0))
            detected_fingerprints = (detected_fingerprints > 0).long()
        
            #print(detected_fingerprints)
            bitwise_accuracy += (detected_fingerprints == fingerprint).float().mean(dim=1).sum().item()

            img_noise_path = os.path.join("/media/giacomo/hdd_ubuntu/gau_blurring_size_1-73_style2_50k", f"{k}")
            os.makedirs(img_noise_path , exist_ok=True)
            png_filename = os.path.join(img_noise_path, filename)
            PIL.Image.fromarray(img_blurred_rgb_array, "RGB").save(png_filename)
            
            """
            usefull to visualize what we are doing. Use it only with few images to try the code

            cv2.imshow("original image", img)
            cv2.waitKey(0)
            cv2.imshow("image with noise", img_noised)
            cv2.waitKey(0)

            img_path_saved = os.path.join("/media/giacomo/hdd_ubuntu/gau_noise_std_0-100_style2_50k", filename)
            img_saved = cv2.imread(img_path_saved,3)

            cv2.imshow("saved image", img_saved)
            cv2.waitKey(0)
            
            """
            
    psnr = main(image_directory, img_noise_path)
    psnr_array.append(psnr)
    size_array.append(i)
    bitwise_accuracy = bitwise_accuracy/j
    accuracy_array.append(bitwise_accuracy)
    

print(size_array)
print(accuracy_array)
print(psnr_array)

plotting(size_array,accuracy_array,psnr_array,"kernel size","Bitwise accuracy","PSNR (dB)","Gaussian blurring")

"""
plt.plot(size_array, accuracy_array, marker='s', linestyle='--', color='black', markerfacecolor='red', markeredgecolor='red')
plt.grid(color='grey', linestyle='-', linewidth=0.5)

plt.yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0]) #to fix the y scale but it can be used also accuracy_array
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]) #to fix the x scale

#figure, axis = plt.subplots(1, 1)
#figure.suptitle("Gaussian noise")
#axis.plot(std_array, accuracy_array)
plt.title("Gaussian blurring", fontweight="bold")
plt.ylabel("Bitwise accuracy")
plt.xlabel("Kernel size")
plt.show()
"""
