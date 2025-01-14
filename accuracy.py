from models import StegaStampDecoder
import torch

import torchvision.transforms as transforms
import torchvision
import numpy as np

import torch
from torchvision.utils import save_image
import os
import argparse
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob
import PIL

parser = argparse.ArgumentParser()

transform = transforms.Compose([
    transforms.ToTensor()  # Convert the image to a tensor
])

class CustomImageFolder():
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir #path with the dataset for the training specified via CLI
        self.filenames = glob.glob(os.path.join(data_dir, "*.png")) #to get all the png image's paths 
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg"))) #to add all the jpeg images' path 
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg"))) #to add all the jpg images' path 
        self.filenames = sorted(self.filenames) #order the file name in ascendent order
        self.transform = transform

    #return the image at the specified index
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)

#insert the path of the images that you want to perturbate
image_directory=""


#insert the path of the decoder that you want to use
dec_path = ""

def main(image_directory, dec_path, decoder=None):

    #Set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                                0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                                0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0]).to(device) #fingerprint embedded in the images
                                


    args = parser.parse_args()

    IMAGE_RESOLUTION = 128
    IMAGE_CHANNELS = 1


    FINGERPRINT_SIZE = len(fingerprint)



    RevealNet_pre = StegaStampDecoder( #decoder and parameters passing
            IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
        )
    RevealNet_pre.load_state_dict(torch.load(dec_path))
    RevealNet_pre = RevealNet_pre.to(device)
    RevealNet_pre.eval()

    if decoder != None:
        RevealNet_pre = decoder


    dataset = CustomImageFolder(image_directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    bitwise_accuracy = 0;

    j=0
    for images, _ in tqdm(dataloader):
       
        y_channel_list = []

        for image in images:
            image = image.permute(1, 2, 0).cpu().numpy()
            
            
            image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
           
            y_channel, u_channel, v_channel = cv2.split(image)
            image = y_channel
            image = torch.from_numpy(image).unsqueeze(0)

            
            y_channel_list.append(image)
        
        images_y_batch = torch.stack(y_channel_list).to(device)

        detected_fingerprints = RevealNet_pre(images_y_batch)
        detected_fingerprints = (detected_fingerprints > 0).long()

        
        for i in enumerate(detected_fingerprints):
            j = j + 1
            #to calculate the accuracy in retrieving the fingerprint (eventually perturbated)
            bitwise_accuracy += (detected_fingerprints[i].detach() == fingerprint).float().mean().sum().item()

    bitwise_accuracy = bitwise_accuracy/j
    print(bitwise_accuracy)
    return bitwise_accuracy
       
if __name__ == "__main__": 
    main(image_directory, dec_path)