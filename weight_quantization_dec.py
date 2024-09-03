import numpy as np
import os
from PIL import Image
import torch
from models import StegaStampDecoder
import matplotlib.pyplot as plt


#In the article this is applied to the ProGAN network and images generation.
#I use StyleGAN2 instead of ProGAN because the images generated with it showed a better bitwise_accuracy.
#However, in the article this robustness study is applied to the network model, but
#in my opinion is more worth to study how this affect the decoder, in order to study
#how the quantization precision can affect it in a malicious point of view.


#Function to quantize model weights to a specific precision
def quantize_weights(model, precision):
    for param in model.parameters():
        if param.requires_grad:
            param.data = torch.round(param.data / precision) * precision


accuracy_array = []
quant_prec_array = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

decoder_path = "/home/giacomo/Desktop/enc_dec_pretrained_celeba/dec.pth"

#fingerprint embedded in the images
fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0])

image_directory = '/media/giacomo/hdd_ubuntu/stylegan2_gen'


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

i = 10**-5

end = 1

step = 10


while i <= end:

    j=0
    for filename in os.listdir(image_directory):

        j += 1 #to count the number of images in the folder

        if j == 11:
            break;
        
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):

            print("Precision")
            print(i)
            

            RevealNet = StegaStampDecoder( #decoder and parameter passing
                IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
            )

            state_dict = torch.load(decoder_path, map_location=device)
            RevealNet.load_state_dict(state_dict)
            RevealNet.to(device)  # Move the model to the device
            RevealNet.eval()      # Set the model to evaluation mode

            quantize_weights(RevealNet, i)

            img_path = os.path.join(image_directory, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            image_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().to(device)

            detected_fingerprints = RevealNet(image_tensor.unsqueeze(0))
            detected_fingerprints = (detected_fingerprints > 0).long()
        
            print(detected_fingerprints)
            bitwise_accuracy += (detected_fingerprints == fingerprint).float().mean(dim=1).sum().item()

            img_array = np.array(img)
        
            print("img_array")
            print(img_array)
            
    quant_prec_array.append(i)
    bitwise_accuracy = bitwise_accuracy/j
    accuracy_array.append(bitwise_accuracy)

    i = i*step
    

print(quant_prec_array)
print(accuracy_array)

plt.plot(quant_prec_array, accuracy_array, marker='s', linestyle='--', color='black', markerfacecolor='red', markeredgecolor='red')
plt.grid(color='grey', linestyle='-', linewidth=0.5)

plt.yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0]) #to fix the y scale but it can be used also accuracy_array
plt.xticks(np.logspace(-5, 0, 6))
plt.xscale("log")

plt.title("Model quantization", fontweight="bold")
plt.ylabel("Bitwise accuracy")
plt.xlabel("Quantization decimal precision")
plt.show()
