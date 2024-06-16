import torch
from torch import nn
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from training_utils import train_utils
import torch.nn.functional as F
from torch.autograd import Variable
from model_utils.spectral import SpectralNorm


# 1 layer
class CNN_k5_single_layer_relu_mp_thresh_quantized(nn.Module):
    def __init__(self, params_model, cnn_method, cnn_output_quantization, max_pool):
        super(CNN_k5_single_layer_relu_mp_thresh_quantized, self).__init__()
        k_size = params_model["kernel_size"]
        dilation = 1
        stride = 1
        groups = 1
        bias = False
        self.weight_quantize = cnn_method
        self.output_quantize = cnn_output_quantization
        self.cnn_method = params_model["cnn_method"]
        P1 = ( stride * (1-1)- 1 + dilation*(k_size - 1))//2 + 1

        self.thresh1 = nn.Parameter(torch.rand(1))
        self.thresh2 = nn.Parameter(torch.rand(1))
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=k_size, stride=stride, padding=P1, bias=bias)
        self.conv1post = nn.Sequential(
            nn.ReLU(),
            torch.nn.MaxPool2d(4),
        )

    def forward(self, x, print_intermediates=False, save_dir=None, min_max=(0.0,0.0), input_white=5, store=0):
        self.conv1.weight.data = self.weight_quantize.apply(self.conv1.weight)
        binarized_x = self.output_quantize.apply(x)

        conv1_out = self.conv1(binarized_x)                         # should be -25 to 25
        conv1post_out = self.conv1post(conv1_out)  # 4,16,16,16     # should be 0 to 25
            
        out = self.output_quantize.apply(conv1post_out-self.thresh1)    
        out_flat = train_utils.scamp_shape(out)
        # print("pre reshaping", out.size())
        row1 = torch.cat((out[:,0:1,:,:],out[:,1:2,:,:],out[:,2:3,:,:],out[:,3:4,:,:]), 2)
        row2 = torch.cat((out[:,4:5,:,:],out[:,5:6,:,:],out[:,6:7,:,:],out[:,7:8,:,:]), 2)
        row3 = torch.cat((out[:,8:9,:,:],out[:,9:10,:,:],out[:,10:11,:,:],out[:,11:12,:,:]), 2)
        row4 = torch.cat((out[:,12:13,:,:],out[:,13:14,:,:],out[:,14:15,:,:],out[:,15:16,:,:]), 2)
        out = torch.cat((row1,row2,row3,row4),3) # 4,1,64,64
        
        if store>0:

            file_path = f'/home/haleyso/cnn_in{store}.txt'
            np.savetxt(file_path, x.view(-1).cpu().detach().numpy())
            
            file_path = f'/home/haleyso/cnn_outs{store}.txt'
            np.savetxt(file_path, conv1post_out.view(-1).cpu().detach().numpy())
            
        # hidden_x = hidden.cpu().detach().numpy() 
        # hidden_x = 255*(hidden_x+1)/2
        # unique_values, counts = np.unique(out.cpu().detach().numpy(), return_counts=True)
        # print(unique_values, counts.min()/(counts.max()+counts.min()))

        # for value, count in zip(unique_values, counts):
            # print(f"{value}: {count} ", end=", ")

        if print_intermediates:
            vid_min = min_max[0]
            vid_max = min_max[1]
            # print(vid_max)
            # print(vid_min)
            # sys.exit()
            # vid_max = 2.3987302780151367
            # vid_min = -2.019804000854492 


            # torch.ones_like(d1)
            da_weights = self.conv1.weight.data.cpu().detach()
            da_weights = 255*(da_weights+1)/2
            d1 = da_weights[0,:,:,:].squeeze()
            d2 = da_weights[1,:,:,:].squeeze()
            d3 = da_weights[2,:,:,:].squeeze()
            d4 = da_weights[3,:,:,:].squeeze()

            d5 = da_weights[4,:,:,:].squeeze()
            d6 = da_weights[5,:,:,:].squeeze()
            d7 = da_weights[6,:,:,:].squeeze()
            d8 = da_weights[7,:,:,:].squeeze()
            
            d9 = da_weights[8,:,:,:].squeeze()
            d10 = da_weights[9,:,:,:].squeeze()
            d11= da_weights[10,:,:,:].squeeze()
            d12= da_weights[11,:,:,:].squeeze()

            d13 = da_weights[12,:,:,:].squeeze()
            d14 = da_weights[13,:,:,:].squeeze()
            d15 = da_weights[14,:,:,:].squeeze()
            d16 = da_weights[15,:,:,:].squeeze()
            # print(d16.size())
            # sys.exit()
            pad = torch.ones_like(d1)*127
            vpad = torch.ones_like(torch.hstack([d1, pad, d2, pad, d3, pad, d4]))*127
            da_weights = torch.vstack([torch.hstack([d1, pad, d2, pad, d3, pad, d4]),vpad,torch.hstack([d5, pad, d6, pad, d7, pad, d8]),vpad,  torch.hstack([d9, pad, d10, pad, d11, pad, d12]),vpad, torch.hstack([d13, pad, d14, pad, d15, pad, d16])])


            input_x = x.cpu().detach()
            input_x = (input_x -(vid_min))/(vid_max-vid_min) * 255 # input is now in scamp regime 0-255
            
            zero_is = ((0.0 - vid_min)/(vid_max-vid_min)) * 255.0
            # print(zero_is)
            
            bin_x =  binarized_x.cpu().detach() # -1 1
            bin_x = 255*(bin_x + 1)/2 # 0,255

            conv1x = conv1_out.cpu().detach() # -25,25
            conv1x = conv1x*input_white +127
            # conv1x = 255*(conv1x + 25)/50          # 0 255

            # print(conv1x.min(), conv1x.max())
            # sys.exit()
            conv1postx = conv1post_out.cpu().detach() # 0,25
            # conv1postx = 255*conv1postx/25         # 0 255
            conv1postx = conv1postx*input_white + 127
            
            # this means the threshold is thresh*5
            out_flatx = out_flat.cpu().detach() # -1 1
            out_flatx = 255*(out_flatx+ 1)/2     # 0 255

            # print(input_x.max(), bin_x.max(), conv1x.max(), conv1postx.max(), out_flatx.max())
            # /home/haleyso/CNN_CGRU/oct2023/8/Set4_8_0019/0
            homef = ('/').join(save_dir.split('/')[:-3])

            train_utils.ensure_dir(save_dir)
            # /home/haleyso/CNN_CGRU/oct2023/8/Set4_8_0019
            plt.imsave(os.path.join(save_dir,"0_input_x.BMP"), input_x.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"0_binarized_x.BMP"), bin_x.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"1_conv1.BMP"),     train_utils.scamp_shape(conv1x).numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"2_conv1post.BMP"), train_utils.scamp_shape(conv1postx).numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"3_thresholded_quantized.BMP"), out_flatx.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(homef,"convweights.BMP"), da_weights.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
        
        return out_flat

# 2 layer
class CNN_k5_two_layer_relu_mp_thresh_quantized(nn.Module):
    def __init__(self, params_model, cnn_method, cnn_output_quantization, max_pool):
        super(CNN_k5_two_layer_relu_mp_thresh_quantized, self).__init__()
        k_size = 5
        dilation = 1
        stride = 1
        groups = 1
        bias = False
        self.weight_quantize = cnn_method
        self.output_quantize = cnn_output_quantization
        self.cnn_method = params_model["cnn_method"]
        P1 = ( stride * (1-1)- 1 + dilation*(k_size - 1))//2 + 1

        self.thresh1 = nn.Parameter(torch.rand(1))
        self.thresh2 = nn.Parameter(torch.rand(1))
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=k_size, stride=stride, padding=P1, bias=bias)
        self.conv1post = nn.Sequential(
            nn.ReLU(),
            torch.nn.MaxPool2d(4),
        )
        self.conv2 = nn.Conv2d(1, 16, kernel_size=k_size, stride=stride, padding=P1, bias=bias)
        self.conv2post = nn.Sequential(
            nn.ReLU(),
            torch.nn.MaxPool2d(4),
        )

    def forward(self, x, print_intermediates=False, save_dir=None, min_max=(0.0,0.0), input_white=None):

        self.conv1.weight.data = self.weight_quantize.apply(self.conv1.weight)
        self.conv2.weight.data = self.weight_quantize.apply(self.conv2.weight)

        binarized_x = self.output_quantize.apply(x)
        # print(x.dtype, binarized_x.dtype, self.conv1.weight.data.dtype)
        # sys.exit()
        out = self.conv1(binarized_x) 
        # print(out.dtype)
        # sys.exit()
        out = self.conv1post(out)  # 4,16,16,16
        out = self.output_quantize.apply(out-self.thresh1)

        row1 = torch.cat((out[:,0:1,:,:],out[:,1:2,:,:],out[:,2:3,:,:],out[:,3:4,:,:]), 2)
        row2 = torch.cat((out[:,4:5,:,:],out[:,5:6,:,:],out[:,6:7,:,:],out[:,7:8,:,:]), 2)
        row3 = torch.cat((out[:,8:9,:,:],out[:,9:10,:,:],out[:,10:11,:,:],out[:,11:12,:,:]), 2)
        row4 = torch.cat((out[:,12:13,:,:],out[:,13:14,:,:],out[:,14:15,:,:],out[:,15:16,:,:]), 2)
        out = torch.cat((row1,row2,row3,row4),3) # 4,1,64,64

        out = self.conv2(out) 
        out = self.conv2post(out)  # 4,16,16,16
        out = self.output_quantize.apply(out-self.thresh2) 
        
        row1_2 = torch.cat((out[:,0:1,:,:],out[:,1:2,:,:],out[:,2:3,:,:],out[:,3:4,:,:]), 2)
        row2_2 = torch.cat((out[:,4:5,:,:],out[:,5:6,:,:],out[:,6:7,:,:],out[:,7:8,:,:]), 2)
        row3_2 = torch.cat((out[:,8:9,:,:],out[:,9:10,:,:],out[:,10:11,:,:],out[:,11:12,:,:]), 2)
        row4_2 = torch.cat((out[:,12:13,:,:],out[:,13:14,:,:],out[:,14:15,:,:],out[:,15:16,:,:]), 2)
        out = torch.cat((row1_2,row2_2,row3_2,row4_2),3) # 4,1,64,64
        return out  

# 1 layer with noise added to train for SCAMP
class CNN_noise(nn.Module):
    def __init__(self, params_model, cnn_method, cnn_output_quantization, max_pool):
        super(CNN_noise, self).__init__()
        k_size = 5
        dilation = 1
        stride = 1
        groups = 1
        bias = False
        self.weight_quantize = cnn_method
        self.output_quantize = cnn_output_quantization
        self.cnn_method = params_model["cnn_method"]
        P1 = ( stride * (1-1)- 1 + dilation*(k_size - 1))//2 + 1

        # self.thresh1 = nn.Parameter(torch.rand(1))
        # self.thresh2 = nn.Parameter(torch.rand(1))
        self.thresh1 = nn.Parameter(torch.ones(1))
        self.thresh2 = nn.Parameter(torch.ones(1))
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=k_size, stride=stride, padding=P1, bias=bias)
        self.conv1post = nn.Sequential(
            nn.ReLU(),
            torch.nn.MaxPool2d(4),
        )

    def forward(self, x, print_intermediates=False, save_dir=None, min_max=(0.0,0.0), input_white=20, store=0):
        self.conv1.weight.data = self.weight_quantize.apply(self.conv1.weight)
        binarized_x = self.output_quantize.apply(x)

        conv1_out = self.conv1(binarized_x)                         # should be -25 to 25
        conv1post_out = self.conv1post(conv1_out)  # 4,16,16,16     # should be 0 to 25
        noise = conv1post_out + 0.6*torch.randn_like(conv1post_out)

        out = self.output_quantize.apply(noise-self.thresh1)    
        out_flat = train_utils.scamp_shape(out)
        # print("pre reshaping", out.size())
        row1 = torch.cat((out[:,0:1,:,:],out[:,1:2,:,:],out[:,2:3,:,:],out[:,3:4,:,:]), 2)
        row2 = torch.cat((out[:,4:5,:,:],out[:,5:6,:,:],out[:,6:7,:,:],out[:,7:8,:,:]), 2)
        row3 = torch.cat((out[:,8:9,:,:],out[:,9:10,:,:],out[:,10:11,:,:],out[:,11:12,:,:]), 2)
        row4 = torch.cat((out[:,12:13,:,:],out[:,13:14,:,:],out[:,14:15,:,:],out[:,15:16,:,:]), 2)
        out = torch.cat((row1,row2,row3,row4),3) # 4,1,64,64

        # print("post reshaping",out_flat.size(), out.size(), torch.sum(out_flat != out))
        # sys.exit()

        if print_intermediates:
            vid_min = min_max[0]
            vid_max = min_max[1]
            # print(vid_max)
            # print(vid_min)
            # sys.exit()
            # vid_max = 2.3987302780151367
            # vid_min = -2.019804000854492 


            # torch.ones_like(d1)
            da_weights = self.conv1.weight.data.cpu().detach()
            da_weights = 255*(da_weights+1)/2
            d1 = da_weights[0,:,:,:].squeeze()
            d2 = da_weights[1,:,:,:].squeeze()
            d3 = da_weights[2,:,:,:].squeeze()
            d4 = da_weights[3,:,:,:].squeeze()

            d5 = da_weights[4,:,:,:].squeeze()
            d6 = da_weights[5,:,:,:].squeeze()
            d7 = da_weights[6,:,:,:].squeeze()
            d8 = da_weights[7,:,:,:].squeeze()
            
            d9 = da_weights[8,:,:,:].squeeze()
            d10 = da_weights[9,:,:,:].squeeze()
            d11= da_weights[10,:,:,:].squeeze()
            d12= da_weights[11,:,:,:].squeeze()

            d13 = da_weights[12,:,:,:].squeeze()
            d14 = da_weights[13,:,:,:].squeeze()
            d15 = da_weights[14,:,:,:].squeeze()
            d16 = da_weights[15,:,:,:].squeeze()
            # print(d16.size())
            # sys.exit()
            pad = torch.ones_like(d1)*127
            vpad = torch.ones_like(torch.hstack([d1, pad, d2, pad, d3, pad, d4]))*127
            da_weights = torch.vstack([torch.hstack([d1, pad, d2, pad, d3, pad, d4]),vpad,torch.hstack([d5, pad, d6, pad, d7, pad, d8]),vpad,  torch.hstack([d9, pad, d10, pad, d11, pad, d12]),vpad, torch.hstack([d13, pad, d14, pad, d15, pad, d16])])


            input_x = x.cpu().detach()
            input_x = (input_x -(vid_min))/(vid_max-vid_min) * 255 # input is now in scamp regime 0-255
            
            zero_is = ((0.0 - vid_min)/(vid_max-vid_min)) * 255.0
            # print(zero_is)
            
            bin_x =  binarized_x.cpu().detach() # -1 1
            bin_x = 255*(bin_x + 1)/2 # 0,255

            conv1x = conv1_out.cpu().detach() # -25,25
            conv1x = conv1x*input_white +127
            # conv1x = 255*(conv1x + 25)/50          # 0 255

            # print(conv1x.min(), conv1x.max())
            # sys.exit()
            conv1postx = conv1post_out.cpu().detach() # 0,25
            # conv1postx = 255*conv1postx/25         # 0 255
            conv1postx = conv1postx*input_white + 127
            
            # this means the threshold is thresh*5
            out_flatx = out_flat.cpu().detach() # -1 1
            out_flatx = 255*(out_flatx+ 1)/2     # 0 255

            # print(input_x.max(), bin_x.max(), conv1x.max(), conv1postx.max(), out_flatx.max())
            # /home/haleyso/CNN_CGRU/oct2023/8/Set4_8_0019/0
            homef = ('/').join(save_dir.split('/')[:-3])

            train_utils.ensure_dir(save_dir)
            # /home/haleyso/CNN_CGRU/oct2023/8/Set4_8_0019
            plt.imsave(os.path.join(save_dir,"0_input_x.BMP"), input_x.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"0_binarized_x.BMP"), bin_x.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"1_conv1.BMP"),     train_utils.scamp_shape(conv1x).numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"2_conv1post.BMP"), train_utils.scamp_shape(conv1postx).numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"3_thresholded_quantized.BMP"), out_flatx.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(homef,"convweights.BMP"), da_weights.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
        
        return out_flat

# 1 layer with noise added to train for SCAMP -- lips
class CNN_noise_lip(nn.Module):
    def __init__(self, params_model, cnn_method, cnn_output_quantization, max_pool):
        super(CNN_noise_lip, self).__init__()
        k_size = 5
        dilation = 1
        stride = 1
        groups = 1
        bias = False
        self.weight_quantize = cnn_method
        self.output_quantize = cnn_output_quantization
        self.cnn_method = params_model["cnn_method"]
        P1 = ( stride * (1-1)- 1 + dilation*(k_size - 1))//2 + 1

        self.thresh0 = nn.Parameter(torch.ones(1))
        self.thresh1 = nn.Parameter(torch.ones(1))
        self.thresh2 = nn.Parameter(torch.ones(1))
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=k_size, stride=stride, padding=P1, bias=bias)
        self.conv1post = nn.Sequential(
            nn.ReLU(),
            torch.nn.MaxPool2d(4),
        )

    def forward(self, x, print_intermediates=False, save_dir=None, min_max=(0.0,0.0), input_white=20, store=0):
        self.conv1.weight.data = self.weight_quantize.apply(self.conv1.weight)
        binarized_x = self.output_quantize.apply(x- self.thresh0)

        conv1_out = self.conv1(binarized_x)                         # should be -25 to 25
        conv1post_out = self.conv1post(conv1_out)  # 4,16,16,16     # should be 0 to 25
        noise = conv1post_out + 0.6*torch.randn_like(conv1post_out)

        out = self.output_quantize.apply(noise-self.thresh1)    
        out_flat = train_utils.scamp_shape(out)
        # print("pre reshaping", out.size())
        row1 = torch.cat((out[:,0:1,:,:],out[:,1:2,:,:],out[:,2:3,:,:],out[:,3:4,:,:]), 2)
        row2 = torch.cat((out[:,4:5,:,:],out[:,5:6,:,:],out[:,6:7,:,:],out[:,7:8,:,:]), 2)
        row3 = torch.cat((out[:,8:9,:,:],out[:,9:10,:,:],out[:,10:11,:,:],out[:,11:12,:,:]), 2)
        row4 = torch.cat((out[:,12:13,:,:],out[:,13:14,:,:],out[:,14:15,:,:],out[:,15:16,:,:]), 2)
        out = torch.cat((row1,row2,row3,row4),3) # 4,1,64,64

        # print("post reshaping",out_flat.size(), out.size(), torch.sum(out_flat != out))
        # sys.exit()

        if print_intermediates:
            vid_min = min_max[0]
            vid_max = min_max[1]
            # print(vid_max)
            # print(vid_min)
            # sys.exit()
            # vid_max = 2.3987302780151367
            # vid_min = -2.019804000854492 


            # torch.ones_like(d1)
            da_weights = self.conv1.weight.data.cpu().detach()
            da_weights = 255*(da_weights+1)/2
            d1 = da_weights[0,:,:,:].squeeze()
            d2 = da_weights[1,:,:,:].squeeze()
            d3 = da_weights[2,:,:,:].squeeze()
            d4 = da_weights[3,:,:,:].squeeze()

            d5 = da_weights[4,:,:,:].squeeze()
            d6 = da_weights[5,:,:,:].squeeze()
            d7 = da_weights[6,:,:,:].squeeze()
            d8 = da_weights[7,:,:,:].squeeze()
            
            d9 = da_weights[8,:,:,:].squeeze()
            d10 = da_weights[9,:,:,:].squeeze()
            d11= da_weights[10,:,:,:].squeeze()
            d12= da_weights[11,:,:,:].squeeze()

            d13 = da_weights[12,:,:,:].squeeze()
            d14 = da_weights[13,:,:,:].squeeze()
            d15 = da_weights[14,:,:,:].squeeze()
            d16 = da_weights[15,:,:,:].squeeze()
            # print(d16.size())
            # sys.exit()
            pad = torch.ones_like(d1)*127
            vpad = torch.ones_like(torch.hstack([d1, pad, d2, pad, d3, pad, d4]))*127
            da_weights = torch.vstack([torch.hstack([d1, pad, d2, pad, d3, pad, d4]),vpad,torch.hstack([d5, pad, d6, pad, d7, pad, d8]),vpad,  torch.hstack([d9, pad, d10, pad, d11, pad, d12]),vpad, torch.hstack([d13, pad, d14, pad, d15, pad, d16])])


            input_x = x.cpu().detach()
            input_x = (input_x -(vid_min))/(vid_max-vid_min) * 255 # input is now in scamp regime 0-255
            
            zero_is = ((0.0 - vid_min)/(vid_max-vid_min)) * 255.0
            # print(zero_is)
            
            bin_x =  binarized_x.cpu().detach() # -1 1
            bin_x = 255*(bin_x + 1)/2 # 0,255

            conv1x = conv1_out.cpu().detach() # -25,25
            conv1x = conv1x*input_white +127
            # conv1x = 255*(conv1x + 25)/50          # 0 255

            # print(conv1x.min(), conv1x.max())
            # sys.exit()
            conv1postx = conv1post_out.cpu().detach() # 0,25
            # conv1postx = 255*conv1postx/25         # 0 255
            conv1postx = conv1postx*input_white + 127
            
            # this means the threshold is thresh*5
            out_flatx = out_flat.cpu().detach() # -1 1
            out_flatx = 255*(out_flatx+ 1)/2     # 0 255

            # print(input_x.max(), bin_x.max(), conv1x.max(), conv1postx.max(), out_flatx.max())
            # /home/haleyso/CNN_CGRU/oct2023/8/Set4_8_0019/0
            homef = ('/').join(save_dir.split('/')[:-3])

            train_utils.ensure_dir(save_dir)
            # /home/haleyso/CNN_CGRU/oct2023/8/Set4_8_0019
            plt.imsave(os.path.join(save_dir,"0_input_x.BMP"), input_x.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"0_binarized_x.BMP"), bin_x.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"1_conv1.BMP"),     train_utils.scamp_shape(conv1x).numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"2_conv1post.BMP"), train_utils.scamp_shape(conv1postx).numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"3_thresholded_quantized.BMP"), out_flatx.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(homef,"convweights.BMP"), da_weights.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
        
        return out_flat