
from math import log10, sqrt 
import cv2 
import numpy as np 
import os
from torchvision import transforms
import PIL



original_image_directory = ''
finger_image_directory = ''



#compressed or fingerprinted or generated
def PSNR(original, compressed): 
   
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal. 
                   
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
     
    for filename in os.listdir(finger_image_directory):

        j = j+1
        
        #to garantee that the "original" dataset has the same png extension as the other dataset
        #it can be changed depending on your needs
        base_name, _ = os.path.splitext(filename)
        filename = base_name + "." + "png"
        
        
        ori_img_path = os.path.join(original_image_directory, filename)
        original = cv2.imread(ori_img_path,3)
        
        fin_img_path = os.path.join(finger_image_directory, filename)
        fingerprinted = cv2.imread(fin_img_path, 3)
        

        #to use only if the image has a size different from 128x128
        
        #original = PIL.Image.fromarray(original)
        #original = transform(original) 

        original = np.array(original)
        
        PSNR_value = PSNR_value + PSNR(original, fingerprinted) 
        

    PSNR_value = PSNR_value/(j)
    print(f"PSNR value is {PSNR_value} dB") 
    return PSNR_value


def main_j(original_image_directory, finger_image_directory): 
    
    transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
            ]
        )

    j=0
    PSNR_value = 0
     
    for filename in os.listdir(finger_image_directory):

        j = j+1
        
        #to garantee that the "original" dataset has the same png extension as the other dataset
        #it can be changed depending on your needs
        base_name, _ = os.path.splitext(filename)
        filename = base_name + "." + "png"
        
        
        ori_img_path = os.path.join(original_image_directory, filename)
        original = cv2.imread(ori_img_path,3)
        
        
        
        base_name, _ = os.path.splitext(filename)
        fin_filename = base_name + "." + "jpg"
        filename = fin_filename
        fin_img_path = os.path.join(finger_image_directory, filename)
        fingerprinted = cv2.imread(fin_img_path, 3)
        
        fin_img_path = os.path.join(finger_image_directory, filename)
        fingerprinted = cv2.imread(fin_img_path, 3)
        

        #to use only if the image has a size different from 128x128
        
        #original = PIL.Image.fromarray(original)
        #original = transform(original) 

        original = np.array(original)
        
        PSNR_value = PSNR_value + PSNR(original, fingerprinted) 
        

    PSNR_value = PSNR_value/(j)
    print(f"PSNR value is {PSNR_value} dB") 
    return PSNR_value

   
       
if __name__ == "__main__": 
    main(original_image_directory, finger_image_directory) 