
from math import log10, sqrt 
import cv2 
import numpy as np 
import os
from torchvision import transforms
import PIL

#original_image_directory = '/media/giacomo/hdd_ubuntu/old/celeba_fin_old_200k'
#finger_image_directory = '/media/giacomo/hdd_ubuntu/new/celeba_fin_new_200k'
#finger_image_directory = '/media/giacomo/hdd_ubuntu/celeba-fingerprinted-200k'
#finger_image_directory = '/media/giacomo/hdd_ubuntu/dataset_celeba/img_celeba'


#compressed or fingerprinted or generated
def PSNR(original, compressed): 
    #print("Original")
    #print(original)

    #print("Fingerprinted")
    #print(compressed)
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0 #color image
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 
  
def main(original_image_directory, finger_image_directory): 
    
    transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
            ]
        )

    j=0
    PSNR_value = 0
     
    for filename in os.listdir(original_image_directory):

        j = j+1

        #if j == 11: break;

        ori_img_path = os.path.join(original_image_directory, filename)
        original = cv2.imread(ori_img_path,3)

        
        #to garantee that the "original" dataset has the same png extension as the other dataset
        #it can be changed depending on your needs
        base_name, _ = os.path.splitext(filename)
        filename = base_name + "." + "png"

        #only for jpeg compression
        #base_name, _ = os.path.splitext(filename)
        #fin_filename = base_name + "." + "jpg"
        #filename = fin_filename

        fin_img_path = os.path.join(finger_image_directory, filename)
        fingerprinted = cv2.imread(fin_img_path, 3)

        #to use only if the image has a size different from 128x128
        
        #original = PIL.Image.fromarray(original)
        #original = transform(original) 

        original = np.array(original)
        
        PSNR_value = PSNR_value + PSNR(original, fingerprinted) 
        #print(j)


    PSNR_value = PSNR_value/(j-1)
    return PSNR_value

    print(f"PSNR value is {PSNR_value} dB") 
       
if __name__ == "__main__": 
    main() 