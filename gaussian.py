import numpy as np
import os
import PIL
from PIL import Image
import cv2
import torch
from models import StegaStampDecoder
import matplotlib.pyplot as plt
from graphs import plotting
from psnr import main
import torchvision.transforms as transforms
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.transforms import ToPILImage




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


def rgb_to_yuv(image_rgb):
    # Assicurati che l'immagine sia in formato float
    image_rgb = image_rgb.astype(np.float32)

    # Crea un array per l'immagine YUV
    yuv_image = np.zeros_like(image_rgb)

    # Applica la conversione
    yuv_image[..., 0] = 0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2]  # Y
    yuv_image[..., 1] = -0.14713 * image_rgb[..., 0] - 0.28886 * image_rgb[..., 1] + 0.436 * image_rgb[..., 2]  # U
    yuv_image[..., 2] = 0.615 * image_rgb[..., 0] - 0.51499 * image_rgb[..., 1] - 0.10001 * image_rgb[..., 2]  # V

    return yuv_image

mean = 0
std = 0

accuracy_array = []
std_array = []
psnr_array = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#decoder_path = "/home/giacomo/Desktop/enc_dec_pretrained_celeba/dec.pth"
#decoder_path = "/media/giacomo/volume/old/trained_byme/dec.pth"
decoder_path = "/media/giacomo/volume/test_yuv/primo/checkpoints/dec.pth"

#fingerprint embedded in the images
fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0]).to(device)

#image_directory = '/media/giacomo/hdd_ubuntu/stylegan2_gen_50k'
#image_directory = '/media/giacomo/volume/old/stylegan2_gen_50k_config-e_25'
image_directory = '/media/giacomo/volume/test_yuv/stylegan2_gen_50k_config-e_25'


IMAGE_RESOLUTION = 128
IMAGE_CHANNELS = 1
FINGERPRINT_SIZE = len(fingerprint)

RevealNet = StegaStampDecoder( #decoder and parameter passing
    IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
)


RevealNet.load_state_dict(torch.load(decoder_path))
RevealNet = RevealNet.to(device)  # Move the model to the device
RevealNet.eval()      # Set the model to evaluation mode



dataset = CustomImageFolder(image_directory, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

to_pil = ToPILImage()

for i in range(11):
    #if i ==1:break;
    bitwise_accuracy = 0
    
    j=0
    total=0
    
    
    #m=0
    
    
    images_y_batch = []
    #for filename in os.listdir(image_directory):
    for images, _ in tqdm(dataloader):
        
        if total==11: break;

        y_channel_list = []

        for image in images:
            #print(images.shape)
            if total==11: break;
            
            print(image.shape)

            image = image.permute(1, 2, 0).cpu().numpy()
            #print(image.shape)
           #noise = torch.normal(mean=mean, std=std, size=image.shape)

            x, y, channels = image.shape  # Include the third dimension for color channels

            # Generate noise with the same shape as that of the image
            noise = np.random.normal(loc=mean, scale=std, size=(x, y, channels))
          
           # print(noise)
            
            image_noised = image + noise
            
            #print(image_noised.shape)
           

            img_noise_path = os.path.join("/media/giacomo/volume/test_yuv/robustness/gau_noise_std_0-100_style2_25_50k", f"{std}") 
            os.makedirs(img_noise_path, exist_ok=True)
            png_filename = os.path.join(img_noise_path,f"image{j}.png" )
            
            # Convert back to tensor, permute to (C, H, W), and save
            image_noised_tensor = torch.from_numpy(image_noised).permute(2, 0, 1)  # Convert back to tensor and permute to (C, H, W)
           # print(image_noised_tensor.shape)
            #image_noised_tensor = image_noised_tensor.float().to(device)  # Move to device if needed

            to_pil(image_noised_tensor).save(png_filename)

             # Clip the pixel values to be between 0 and 255 and convert to uint8
            #image_noised = image_noised*255
            #image_noised = np.clip(image_noised, 0, 255).astype(np.float)
           

            # Converti l'immagine RGB in YUV usando OpenCV
            image_noised_yuv = rgb_to_yuv(image_noised)
            #image_noised = image_noised/255
            y_channel, u_channel, v_channel = cv2.split(image_noised)
            #print("Valore y")
            #print(y_channel)
            y_channel = torch.from_numpy(y_channel).float().unsqueeze(0).to(device)
            y_channel = y_channel
            #print("shape di image_y_noised")
            #print(image_y_noised.shape)
            y_channel_list.append(y_channel)
            total+=1

            images_y_batch = torch.stack(y_channel_list).to(device)
            #print("shape di batch")
            #print(images_y_batch.shape)

        
        
    
        #print("j")
        #print(j)
        detected_fingerprints = RevealNet(images_y_batch)
        detected_fingerprints = (detected_fingerprints > 0).long()
        #print("shape firme")
        #print(detected_fingerprints.shape)
        
        for l in enumerate(detected_fingerprints):
            
            #print(f"firma{j}")
            #print(detected_fingerprints[l])
            #to calculate the accuracy in retrieving the fingerprint (eventually perturbated)
            bitwise_accuracy += (detected_fingerprints[l].detach() == fingerprint).float().mean().sum().item()
            #print(bitwise_accuracy)
            j = j + 1
            
            
        
            
    #print(img_noise_path)
    psnr = main(image_directory, img_noise_path)
    psnr_array.append(psnr)
    std_array.append(std)
    std +=10
    bitwise_accuracy = bitwise_accuracy/j
    accuracy_array.append(bitwise_accuracy)
    

print(std_array)
print(accuracy_array)
print(psnr_array)

plotting(std_array,accuracy_array,psnr_array,"Noise std","Bitwise accuracy","PSNR (dB)","Gaussian noise")

