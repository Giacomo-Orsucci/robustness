import torch
from models import StegaStampDecoder
import os
from PIL import Image
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#image_directory = "/media/giacomo/hdd_ubuntu/stylegan2_gen_noised_0.1"
image_directory = "/media/giacomo/hdd_ubuntu/prova_gen"
decoder_path = "/home/giacomo/Desktop/enc_dec_pretrained_celeba/dec.pth"
fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0])#trovata a mano sulla base dell'enc pretrained


opposite = torch.tensor([1,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,
                            1,0,1,1,1,1,1,0,0,0,0,0,1,0,0,1,0,1,0,1,0,0,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,
                            1,0,1,0,0,0,1,0,1,0,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1])
IMAGE_RESOLUTION = 128
IMAGE_CHANNELS = 3
FINGERPRINT_SIZE = len(fingerprint)


print("Il problema è il caricamento del decoder")
RevealNet = StegaStampDecoder( #decoder e passaggio dei parametri
    IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
)


print("Using device:", device)
#device = "cpu"
print("Using device:", device)
state_dict = torch.load(decoder_path, map_location=device)
RevealNet.load_state_dict(state_dict)
RevealNet.to(device)  # Move the model to the device
RevealNet.eval()      # Set the model to evaluation mode

bitwise_accuracy = 0
fingerprint = (fingerprint > 0).long().to(device)
opposite = (opposite > 0).long().to(device)
NUM_IMAGES = 11

for filename in os.listdir(image_directory):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(image_directory, filename)
        img = Image.open(img_path)
        
        # Convert the image to a NumPy array
        img_array = np.array(img)
        
        # Normalize the image and convert to tensor
        #img_array = img_array / 255.0  # Normalize to range [0, 1]
        image_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().to(device)

        detected_fingerprints = RevealNet(image_tensor.unsqueeze(0))
        detected_fingerprints = (detected_fingerprints > 0).long()
        
        print(detected_fingerprints)
        bitwise_accuracy += (detected_fingerprints == fingerprint).float().mean(dim=1).sum().item()
        #bitwise_accuracy += (detected_fingerprints == opposite).float().mean(dim=1).sum().item()

bitwise_accuracy = bitwise_accuracy / NUM_IMAGES
print(bitwise_accuracy)

       
        