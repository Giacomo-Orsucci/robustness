import numpy as np
import os
from PIL import Image
import torch
from models import StegaStampDecoder
from accuracy import main as maina
import matplotlib.pyplot as plt



#Function to quantize model weights to a specific precision
def quantize_weights(model, precision):
    for param in model.parameters():
        if param.requires_grad:
            param.data = torch.round(param.data / precision) * precision


accuracy_array = []
quant_prec_array = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#insert the path of the decoder that you want to use
decoder_path = ""


#fingerprint embedded in the images
fingerprint = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,
                            0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,
                            0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0])

#insert the path of the images that you want to perturbate
image_directory = ''


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

i = 10**-5

end = 1

step = 10


while i <= end:

    j=0

    RevealNet = StegaStampDecoder( #decoder and parameter passing
            IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
        )

    state_dict = torch.load(decoder_path, map_location=device)
    RevealNet.load_state_dict(state_dict)
    RevealNet.to(device)  # Move the model to the device
    RevealNet.eval()      # Set the model to evaluation mode

    quantize_weights(RevealNet, i)
                   
    quant_prec_array.append(i)
    bitwise_accuracy = maina(image_directory, decoder_path, RevealNet )
    accuracy_array.append(bitwise_accuracy)

    i = i*step
    

plt.plot(quant_prec_array, accuracy_array, marker='s', linestyle='--', color='black', markerfacecolor='red', markeredgecolor='red')
plt.grid(color='grey', linestyle='-', linewidth=0.5)

plt.yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0]) #to fix the y scale but it can be used also accuracy_array
plt.xticks(np.logspace(-5, 0, 6))
plt.xscale("log")

plt.title("Model quantization", fontweight="bold")
plt.ylabel("Bitwise accuracy")
plt.xlabel("Quantization decimal precision")
plt.show()
