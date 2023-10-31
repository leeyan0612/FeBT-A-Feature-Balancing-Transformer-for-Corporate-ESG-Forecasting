from torch import nn
import torch
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(7, 1)):
        super(ResidualBlock, self).__init__()
        self.res_down_conv1 = nn.Conv2d(channel_in, channel_out, kernel_size, stride=1, padding=(3, 0), dilation=1)
        self.res_down_bn1 = torch.nn.BatchNorm2d(channel_out,
                     eps=1e-05, 
                     momentum=0.1, 
                     affine=True, 
                     track_running_stats=True)        
        self.res_down_conv2 = nn.Conv2d(channel_out, channel_out, kernel_size, stride=1, padding=0, dilation=1)
        self.res_down_bn2 = torch.nn.BatchNorm2d(channel_out,
                     eps=1e-05, 
                     momentum=0.1, 
                     affine=True, 
                     track_running_stats=True)       
        self.res_down_conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, stride=1, padding=0, dilation=1)


        self.act_fnc = nn.LeakyReLU()

    def forward(self, x):
        skip = self.res_down_conv3(x)
        x = self.act_fnc(self.res_down_bn1(self.res_down_conv1(x)))
        x = self.res_down_conv2(x)

        return self.act_fnc(self.res_down_bn2(x + skip))
    
class ResidualUpBlock(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=(3, 1), scale_factor=(2, 1)):
        super(ResidualUpBlock, self).__init__()

        self.res_up_conv1 = nn.Conv2d(channel_in, channel_out, kernel_size, stride=1, padding=(1, 0), dilation=1)
        self.res_up_bn1 = torch.nn.BatchNorm2d(channel_out,
                     eps=1e-05,  
                     affine=True, 
                     momentum=0.1,
                     track_running_stats=True)   
        self.res_up_conv2 = nn.Conv2d(channel_out, channel_out, kernel_size, stride=1, padding=0, dilation=1)
        self.res_up_bn2 = torch.nn.BatchNorm2d(channel_out,
                     eps=1e-05, 
                     momentum=0.1, 
                     affine=True, 
                     track_running_stats=True)
        self.res_up_conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, stride=1, padding=0, dilation=1)

        self.up_nn = nn.Upsample(scale_factor=(2, 1), mode="nearest")
        
        self.act_fnc = nn.LeakyReLU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.res_up_conv3(x)
        x = self.act_fnc(self.res_up_bn1(self.res_up_conv1(x)))
        x = self.res_up_conv2(x)

        return self.act_fnc(self.res_up_bn2(x + skip))
    
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        
        self.hidden1 = ResidualBlock(1, 64)
        self.hidden2 = ResidualBlock(64, 32)
        self.hidden3 = ResidualBlock(32, 16)
        self.hidden4 = ResidualBlock(16, 8)
        self.hidden5 = ResidualBlock(8, 1)
        
        self.mu = nn.Linear(13, 64)
        self.var = nn.Linear(13, 64)

    def reparameterization(self, mu, log_var):
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps 

    def forward(self, x):
        x = x.view(-1, 1, 43, 1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        
        x = x.view(-1, 13)
        
        mu = self.mu(x)  
        log_var = self.var(x)  
        z = self.reparameterization(mu, log_var)
        
        z = z.view(-1, 1, 64, 1)
        
        return z

    
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.hidden1 = ResidualUpBlock(1, 64)
        self.hidden2 = ResidualUpBlock(64, 32)
        self.hidden3 = ResidualUpBlock(32, 1)
        
        self.linear = nn.Linear(1522, 172)
        
    def forward(self, x_hat):
        x_hat = self.hidden1(x_hat)
        x_hat = self.hidden2(x_hat)
        x_hat = self.hidden3(x_hat)
        x_hat = x_hat.view(-1, 1522)

        x_hat = self.linear(x_hat)

        return x_hat




class MAE(nn.Module):
    def __init__(self, epoch, epochs):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, epoch, epochs):
        
        x_label = x[:,129:301]
        z_final = None 
        

        unique_tensor = x_label[::4]
    
        x1 = x[:,0:43]
        x2 = x[:,43:86]
        x3 = x[:,86:129]
        
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z3 = self.encoder(x3)
        
        z = torch.cat((z1, z2, z3), dim=2)
        
        x_hat = self.decoder(z)
        
        if epoch == epochs:
            
            y1 = unique_tensor[:,0:43]
            y2 = unique_tensor[:,43:86]
            y3 = unique_tensor[:,86:129]
            y4 = unique_tensor[:,129:172]
            
            zz1 = self.encoder(y1)
            zz2 = self.encoder(y2)
            zz3 = self.encoder(y3)
            zz4 = self.encoder(y4)
            
            z_final = torch.cat((zz1, zz2, zz3, zz4), dim=2)
            
            z_final = z_final.flatten(1)
            
        return x_hat, z, x_label, z_final