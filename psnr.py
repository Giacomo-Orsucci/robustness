
from math import log10, sqrt 
import cv2 
import numpy as np 
import os
from torchvision import transforms
import PIL

original_image_directory = '/media/giacomo/hdd_ubuntu/dataset_celeba/img_celeba'
finger_image_directory = '/media/giacomo/hdd_ubuntu/celeba-fingerprinted-200k'
#finger_image_directory = '/media/giacomo/hdd_ubuntu/dataset_celeba/img_celeba'


#compressed or fingerprinted or generated
def PSNR(original, compressed): 
    print("Original")
    print(original)

    print("Fingerprinted")
    print(compressed)
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0 #color image
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 
  
def main(): 
    
    transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )

    j=0
    PSNR_value = 0
     
    for filename in os.listdir(original_image_directory):

        j = j+1

        if j == 31:
            break;

        ori_img_path = os.path.join(original_image_directory, filename)
        original = cv2.imread(ori_img_path,3)

        
        #uncomment or modify in case of different extension
        base_name, _ = os.path.splitext(filename)
        filename = base_name + "." + "png"




        fin_img_path = os.path.join(finger_image_directory, filename)
        fingerprinted = cv2.imread(fin_img_path, 3)

        original = PIL.Image.fromarray(original)
        original = transform(original) #to properly resize celeba images

        original = original.permute(1, 2, 0).numpy() * 255  # Reshape to (128, 128, 3) and scale
        

        #fingerprinted = PIL.Image.fromarray(fingerprinted)
        #fingerprinted = transform(fingerprinted) #to properly resize celeba images

        #fingerprinted = fingerprinted.permute(1, 2, 0).numpy() * 255  # Reshape to (128, 128, 3) and scale
        #fingerprinted = PIL.Image.fromarray(fingerprinted)
        #fingerprinted = transform(fingerprinted) #to properly resize celeba images


        PSNR_value = PSNR_value + PSNR(original, fingerprinted) 


    PSNR_value = PSNR_value/(j-1)

    print(f"PSNR value is {PSNR_value} dB") 
       
if __name__ == "__main__": 
    main() 