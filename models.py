import math
import torch
from torch import nn
from torch.nn.functional import relu, sigmoid


class StegaStampEncoder(nn.Module):
    def __init__( #encoder constructor (embeds the fingerprint)
        self,
        resolution=32, #images default resolution (32x32)
        IMAGE_CHANNELS=1, #default color channels. 1 = greyscale
        fingerprint_size=100, #fingerprint dimension
        return_residual=False, #differences beetwen original and encoded images
    ):
        super(StegaStampEncoder, self).__init__()
        self.fingerprint_size = fingerprint_size
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        #option to return the residual images
        self.return_residual = return_residual 
        #layer fully connected, here we have the mapping beetwen the fingerprint and a feature compatible space
        #in other words the fingerprint is got compatible to make it possible to transform in a 2D structure
        #compatible with our images. All this work is done because the fingerprint is embedded in the images, but
        #originally is a vector
        self.secret_dense = nn.Linear(self.fingerprint_size, 16 * 16 * IMAGE_CHANNELS)

        #images resolution must be a power of 2
        log_resolution = int(math.log(resolution, 2))
        assert resolution == 2 ** log_resolution, f"Image resolution must be a power of 2, got {resolution}."

        #the definition of the architecture

        #necessary to match the fingerprint spatial dimension with the network feautures map
        self.fingerprint_upsample = nn.Upsample(scale_factor=(2**(log_resolution-4), 2**(log_resolution-4))) #to rescale the fingerprint dimension to the image resolution
        
        #parameters for the subsampling and the features extraction

        #in order: input channels (doubled for images and fingerprint)
        #          output channels
        #          kernel dimension
        #          stride dimension
        #          padding dimension
        self.conv1 = nn.Conv2d(2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)

        #oversampling and reconstruction
        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1)) #image padding: left, right, top, bot
        self.up6 = nn.Conv2d(256, 128, 2, 1)
        #doubles images height and width 
        self.upsample6 = nn.Upsample(scale_factor=(2, 2)) 
        self.conv6 = nn.Conv2d(128 + 128, 128, 3, 1, 1)

        #more layers for oversampling and reconstruction
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1)) 

        self.up7 = nn.Conv2d(128, 64, 2, 1)
        self.upsample7 = nn.Upsample(scale_factor=(2, 2))
        self.conv7 = nn.Conv2d(64 + 64, 64, 3, 1, 1)
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up8 = nn.Conv2d(64, 32, 2, 1)
        self.upsample8 = nn.Upsample(scale_factor=(2, 2))
        self.conv8 = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up9 = nn.Conv2d(32, 32, 2, 1)
        self.upsample9 = nn.Upsample(scale_factor=(2, 2))
        self.conv9 = nn.Conv2d(32 + 32 + 2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1)

        #final layer with a number of channels in output equal to the initial one
        self.residual = nn.Conv2d(32, IMAGE_CHANNELS, 1)

    #function to make the forward propagation
    def forward(self, fingerprint, image):
        #fully connected layer to transform the fingerprint in a feature map
        #relu applies an activation function that is a non linear ramp. In this way all the negative values 
        #are put to zero. This is necessary to get the network able to learn non linear relation.
        fingerprint = relu(self.secret_dense(fingerprint))
        #the fingerprint is modified in a tensor with its dimension, with the image number of channels and 16x16 dimension
        fingerprint = fingerprint.view((-1, self.IMAGE_CHANNELS, 16, 16))
        fingerprint_enlarged = self.fingerprint_upsample(fingerprint)#fingerprint dimension upscaling
        #oversampling fingerprint and input image concatenation
        #necessary to exctract features from the images and the fingerprint
        inputs = torch.cat([fingerprint_enlarged, image], dim=1)

        #convolutionary layer to exctract features. Relu application and connection
        conv1 = relu(self.conv1(inputs))
        conv2 = relu(self.conv2(conv1))
        conv3 = relu(self.conv3(conv2))
        conv4 = relu(self.conv4(conv3))
        conv5 = relu(self.conv5(conv4))

        #oversampling and merging
        up6 = relu(self.up6(self.pad6(self.upsample6(conv5))))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = relu(self.conv6(merge6))

        
        up7 = relu(self.up7(self.pad7(self.upsample7(conv6))))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = relu(self.conv7(merge7))
        up8 = relu(self.up8(self.pad8(self.upsample8(conv7))))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = relu(self.conv8(merge8))
        up9 = relu(self.up9(self.pad9(self.upsample9(conv8))))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = relu(self.conv9(merge9))
        #the final convolutionary layer is applied and the residual image is generated
        conv10 = relu(self.conv10(conv9))
        residual = self.residual(conv10)
        
        #if the residual image is not returned, we apply a sigmoid function (s shaped )
        #to grant that the output is beetwen 0 and 1
        if not self.return_residual:
            residual = sigmoid(residual)
        return residual

#decoder model
class StegaStampDecoder(nn.Module):
    def __init__(self, resolution=32, IMAGE_CHANNELS=1, fingerprint_size=1):
        super(StegaStampDecoder, self).__init__()
        self.resolution = resolution
        self.IMAGE_CHANNELS = IMAGE_CHANNELS

        #network layer definition. We have a progressive reduction of the spatial dimensions to increase
        #channels number. Every layer has a linear activation function activated
        self.decoder = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, (3, 3), 2, 1),  # 16
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 8
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # 4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 2
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), 2, 1),
            nn.ReLU(),
        )
        #layer fully connected that maps the output in 512 units and after in the fingerprint dimension
        self.dense = nn.Sequential(
            nn.Linear(resolution * resolution * 128 // 32 // 32, 512),
            nn.ReLU(),
            nn.Linear(512, fingerprint_size),
        )
    #decodification and return of the fingerprint
    def forward(self, image):
        x = self.decoder(image)
        x = x.view(-1, self.resolution * self.resolution * 128 // 32 // 32)
        return self.dense(x)

