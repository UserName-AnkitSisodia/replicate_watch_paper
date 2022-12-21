"""
Module containing the main VAE class.
"""
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from models.initialization import weights_init
from models.regression import WTPregression

class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, regression, latent_dim, model_type, threshold_val,sup_signal):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(self.img_size, self.latent_dim)
        self.model_type = model_type
        self.threshold_val = threshold_val
        self.sup_signal = sup_signal

        if self.model_type == 'm1':
           self.regression = regression(17) 
        elif self.model_type == 'm2':
           if self.sup_signal == 'brand':
              self.regression = regression(self.latent_dim,5)
           elif self.sup_signal == 'circa':
              self.regression = regression(self.latent_dim,8)
           elif self.sup_signal == 'movement':
              self.regression = regression(self.latent_dim,3)
           elif self.sup_signal == 'material':
              self.regression = regression(self.latent_dim,4)
           elif self.sup_signal == 'location':
              self.regression = regression(self.latent_dim,4)
           elif self.sup_signal == 'price':
              self.regression = regression(self.latent_dim)
        elif self.model_type == 'm3':
           self.regression = regression(17+self.latent_dim)
        elif self.model_type == 'm4a':
           self.regression = regression(20+self.latent_dim) ## without Amsterdam & London
        elif self.model_type == 'm4c':
           self.regression = regression(18+self.latent_dim)
        elif self.model_type == 'm5b':
           self.regression = regression(21+self.latent_dim) ## without Amsterdam & London
        elif self.model_type == 'm6a':
           self.regression = regression(18+self.latent_dim*2)
        elif self.model_type == 'm6b':
           self.regression = regression(21+self.latent_dim*2) ## without Amsterdam & London
        elif self.model_type == 'm7a':
           self.regression = regression(19+self.latent_dim*2)
        elif self.model_type == 'm7b':
           self.regression = regression(22+self.latent_dim*2) ## without Amsterdam & London
        elif self.model_type == 'm8':
           self.regression = regression(20) ## without Amsterdam & London
        elif self.model_type == 'm9':
           self.regression = regression(21) ## without Amsterdam & London
        elif self.model_type == 'm10':
           self.regression = regression(22) ## without Amsterdam & London

        self.decoder = decoder(self.img_size, self.latent_dim)

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def meaningful_visual_attributes(self, mean, logvar, threshold_val):
        """
        """
        latent_dim = mean.size(1)
        batch_size = mean.size(0)
        latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
        zeros = torch.zeros([mean.size(0),mean.size(1)])
        ones = torch.ones([mean.size(0),mean.size(1)])

        for i in range(latent_dim):
            if latent_kl[i].item() < threshold_val:
                ones[:,i] = zeros[:,i]

        return ones

    def forward(self, x, location, brand,circa,movement,diameter,material,timetrend):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        b_s = diameter.shape[0]
        tt_s = timetrend.shape[0]

        diameter = torch.reshape(diameter,(b_s,1))
        timetrend = torch.reshape(timetrend,(tt_s,1))

        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        visual_attributes = self.meaningful_visual_attributes(*latent_dist, self.threshold_val)

        if self.model_type == 'm1':
           wtp_pred = self.regression(torch.hstack((brand.cuda(),circa.cuda(),movement.cuda(),diameter.float().cuda(),material.cuda())))
        elif self.model_type == 'm2':
           wtp_pred = self.regression(torch.mul(latent_dist[0],visual_attributes.cuda()))
        elif self.model_type == 'm3':
           wtp_pred = self.regression(torch.hstack((torch.mul(latent_dist[0],visual_attributes.cuda()),brand.cuda(),circa.cuda(),movement.cuda(),diameter.float().cuda(),material.cuda())))
        elif self.model_type == 'm4a':
           wtp_pred = self.regression(torch.hstack((torch.mul(latent_dist[0],visual_attributes.cuda()),location.cuda(),brand.cuda(),circa.cuda(),movement.cuda(),diameter.float().cuda(),material.cuda())))
        elif self.model_type == 'm4c':
           wtp_pred = self.regression(torch.hstack((torch.mul(latent_dist[0],visual_attributes.cuda()),brand.cuda(),circa.cuda(),movement.cuda(),diameter.float().cuda(),material.cuda(),timetrend.float().cuda())))
        elif self.model_type == 'm5b':
           wtp_pred = self.regression(torch.hstack((torch.mul(latent_dist[0],visual_attributes.cuda()),location.cuda(),brand.cuda(),circa.cuda(),movement.cuda(),diameter.float().cuda(),material.cuda(),timetrend.float().cuda())))
        elif self.model_type == 'm6a':
           wtp_pred = self.regression(torch.hstack((torch.mul(latent_dist[0],visual_attributes.cuda()),brand.cuda(),circa.cuda(),movement.cuda(),diameter.float().cuda(),material.cuda(),timetrend.float().cuda(),torch.matmul(torch.diag(torch.flatten(timetrend.float().cuda())),torch.mul(latent_dist[0],visual_attributes.cuda())))))
        elif self.model_type == 'm6b':
           wtp_pred = self.regression(torch.hstack((torch.mul(latent_dist[0],visual_attributes.cuda()),location.cuda(),brand.cuda(),circa.cuda(),movement.cuda(),diameter.float().cuda(),material.cuda(),timetrend.float().cuda(),torch.matmul(torch.diag(torch.flatten(timetrend.float().cuda())),torch.mul(latent_dist[0],visual_attributes.cuda())))))
        elif self.model_type == 'm7a':
           wtp_pred = self.regression(torch.hstack((torch.mul(latent_dist[0],visual_attributes.cuda()),brand.cuda(),circa.cuda(),movement.cuda(),diameter.float().cuda(),material.cuda(),timetrend.float().cuda(),torch.matmul(torch.diag(torch.flatten(timetrend.float().cuda())),torch.mul(latent_dist[0],visual_attributes.cuda())),torch.mul(timetrend.float().cuda(),timetrend.float().cuda()))))
        elif self.model_type == 'm7b':
           wtp_pred = self.regression(torch.hstack((torch.mul(latent_dist[0],visual_attributes.cuda()),location.cuda(),brand.cuda(),circa.cuda(),movement.cuda(),diameter.float().cuda(),material.cuda(),timetrend.float().cuda(),torch.matmul(torch.diag(torch.flatten(timetrend.float().cuda())),torch.mul(latent_dist[0],visual_attributes.cuda())),torch.mul(timetrend.float().cuda(),timetrend.float().cuda()))))
        elif self.model_type == 'm8':
           wtp_pred = self.regression(torch.hstack((location.cuda(),brand.cuda(),circa.cuda(),movement.cuda(),diameter.float().cuda(),material.cuda())))
        elif self.model_type == 'm9':
           wtp_pred = self.regression(torch.hstack((location.cuda(),brand.cuda(),circa.cuda(),movement.cuda(),diameter.float().cuda(),material.cuda(),timetrend.float().cuda())))
        elif self.model_type == 'm10':
           wtp_pred = self.regression(torch.hstack((location.cuda(),brand.cuda(),circa.cuda(),movement.cuda(),diameter.float().cuda(),material.cuda(),timetrend.float().cuda(),torch.mul(timetrend.float().cuda(),timetrend.float().cuda()))))

        return reconstruct, latent_dist, latent_sample, wtp_pred, visual_attributes

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample

#Residual down sampling block for the encoder
#Average pooling is used to perform the downsampling
class Res_down(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 2):
        super(Res_down, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale,scale)

    def forward(self, x):
        skip = self.conv3(self.AvePool(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x

#Residual up sampling block for the decoder
#Nearest neighbour is used to perform the upsampling
class Res_up(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 2):
        super(Res_up, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        
        self.UpNN = nn.Upsample(scale_factor = scale,mode = "nearest")
        
    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x

class Encoder_VGG(nn.Module):
    def __init__(self,
                 img_size,
                 latent_dim=10):
        super(Encoder_VGG, self).__init__()

        channels = 3
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256*2
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.conv1 = Res_down(channels, hid_channels)#64
        self.conv2 = Res_down(hid_channels, 2*hid_channels)#32
        self.conv3 = Res_down(2*hid_channels, 4*hid_channels)#16
        self.conv4 = Res_down(4*hid_channels, 8*hid_channels)#8
        self.conv5 = Res_down(8*hid_channels, 8*hid_channels)#4
        self.conv_mu = nn.Conv2d(8*hid_channels, latent_dim, 2, 2)#2
        self.conv_logvar = nn.Conv2d(8*hid_channels, latent_dim, 2, 2)#2

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        return mu, logvar

class Decoder_VGG(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        super(Decoder_VGG, self).__init__()
        channels = 3
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256*2
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size
        self.conv1 = Res_up(latent_dim, hid_channels*8)
        self.conv2 = Res_up(hid_channels*8, hid_channels*8)
        self.conv3 = Res_up(hid_channels*8, hid_channels*4)
        self.conv4 = Res_up(hid_channels*4, hid_channels*2)
        self.conv5 = Res_up(hid_channels*2, hid_channels)
        self.conv6 = Res_up(hid_channels, hid_channels//2)
        self.conv7 = nn.Conv2d(hid_channels//2, channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x 

class Encoder(nn.Module):
    def __init__(self,
                 img_size,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256*2 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        """
        super(Encoder, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256*2
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.conv_128=nn.Conv2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = torch.nn.functional.leaky_relu(self.conv_64(x))
        x = torch.nn.functional.leaky_relu(self.conv_128(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.nn.functional.leaky_relu(self.lin1(x))
        x = torch.nn.functional.leaky_relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256*2 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        """
        super(Decoder, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256*2
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        self.convT_128 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, stride=2, padding=1, dilation=1)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.nn.functional.leaky_relu(self.lin1(z))
        x = torch.nn.functional.leaky_relu(self.lin2(x))
        x = torch.nn.functional.leaky_relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        x = torch.nn.functional.leaky_relu(self.convT_128(x))
        x = torch.nn.functional.leaky_relu(self.convT_64(x))
        x = torch.nn.functional.leaky_relu(self.convT1(x))
        x = torch.nn.functional.leaky_relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x

