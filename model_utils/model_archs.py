import torch
from torch import nn
import numpy as np
from model_utils import cnn, rnn
import sys
import os
from training_utils import train_utils, weight_drop

''' architectures
- CNN_RNN_linear
- CNN_linear
- RNN_linear

- CNN_RNN_linear_linear
- CNN_linear_linear
'''



class CNN_RNN_linear(nn.Module):
    def __init__(self, cnn_model, cnn_params, rnn_model, rnn_params, final_conv_method, num_classes, max_pool=1, eight_channels=False, bfloat=False, dropout=None):
        super(CNN_RNN_linear, self).__init__()

        self.cnn_method = cnn_params["cnn_method"]
        self.rnn_method = rnn_params["method"]
        self.final_cnn_method = cnn_params["final_conv_method"]
        self.quantize = final_conv_method
        self.cnn_model = cnn_model
        self.rnn_model = rnn_model
        self.thresh3 = nn.Parameter(torch.zeros(1))


        self.fin_conv = nn.Conv2d(rnn_params['rnn_hidden_size'], 1, kernel_size=1, stride=1, dilation=1, padding=0, bias=False)
        self.fin_convpost = nn.Sequential(
            nn.BatchNorm2d(1),
            torch.nn.MaxPool2d(max_pool),
            nn.Tanh(),
        )
        self.im_out_size = int(64/max_pool)
        if eight_channels:
            self.im_out_size_other = int(64/max_pool/2)
        else:
            self.im_out_size_other = int(64/max_pool)

        if dropout==None:
            self.fc = nn.Linear(self.im_out_size*self.im_out_size_other, num_classes)
        else:
            self.fc = weight_drop.WeightDropLinear(self.im_out_size*self.im_out_size_other, num_classes, weight_dropout=dropout)

        self.bfloat = bfloat

    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0), input_white=5):
        if self.bfloat:
            input = input.to(dtype=torch.bfloat16)

        if (not linear_only):
            if input.dim() <5:
                i = torch.unsqueeze(input, 1)
            else:
                i = input
            b_z, ts, c, h, w = i.shape
            ii = 0
            h_next        = None
            
            
            if print_intermediates:
                save_path = os.path.join(save_dir, str(ii))
                train_utils.ensure_dir(save_path)
                # print(save_path)
                # sys.exit()
            else:
                save_path = None
            
            store = 1

            h_next, out_next = self.rnn_model(self.cnn_model(input[:,ii], print_intermediates, save_path, min_max, input_white, store=store), h_next, print_intermediates, save_path, input_white, store=store)        # 4, 1, 64, 64
            
            for ii in range(1, ts):
                store +=1
                if print_intermediates:
                    save_path = os.path.join(save_dir, str(ii))
                    train_utils.ensure_dir(save_path)
                    
                h_next, out_next = self.rnn_model(self.cnn_model(input[:,ii], print_intermediates, save_path, min_max, input_white, store=store), h_next, print_intermediates, save_path, input_white, store=store)

            # can we get rid of the final conv?
            self.fin_conv.weight.data = self.quantize.apply(self.fin_conv.weight)
            out =  self.fin_conv(out_next)
            out =  self.fin_convpost(out)
            out = out-self.thresh3   # 4, 1, 64, 64
            out = self.fc(out.view(out.size(0), -1)) 
        
        else: # run linear only
            self.fin_conv.weight.data = self.quantize.apply(self.fin_conv.weight)
            out =  self.fin_conv(input)
            out =  self.fin_convpost(out)
            out = out-self.thresh3   # 4, 1, 64, 64
            out = self.fc(out.view(out.size(0), -1))  
        return out


class CNN_linear(nn.Module):
    def __init__(self, cnn_model, cnn_params, final_conv_method, num_classes, max_pool=1, eight_channels=False):
        super(CNN_linear, self).__init__()

        self.quantize = final_conv_method
        self.cnn_model = cnn_model
        self.thresh3 = nn.Parameter(torch.zeros(1))
        self.max_pool = max_pool
        self.bias = cnn_params["bias"]
        self.in_size = 64 # 16 for bose2020
        self.in_size = 1 # for so_2022_batch_norm
        self.fin_conv = nn.Conv2d(self.in_size, 1, kernel_size=1, stride=1, dilation=1, padding=0, bias=self.bias)
        self.fin_conv_post = nn.Sequential(
            nn.BatchNorm2d(1),
            torch.nn.MaxPool2d(max_pool),
            nn.Tanh(),
        )
        self.im_out_size = int(64/self.max_pool)
        self.fc = nn.Linear(self.im_out_size*self.im_out_size, num_classes)

    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0), input_white=None):

        out = self.cnn_model(input)        # 4, 1, 64, 64
        self.fin_conv.weight.data = self.quantize.apply(self.fin_conv.weight)
        out =  self.fin_conv(out)
  
        out =  self.fin_conv_post(out)
        out = out - self.thresh3
        out = self.fc(out.view(out.size(0), -1)) 
        return out



class RNN_linear(nn.Module):
    def __init__(self, rnn_model, rnn_params, final_conv_method, num_classes, max_pool=1, eight_channels=False):
        super(RNN_linear, self).__init__()

        self.quantize = final_conv_method
        self.rnn_model = rnn_model
        self.thresh3 = nn.Parameter(torch.zeros(1))

        self.fin_conv = nn.Conv2d(rnn_params['rnn_hidden_size'], 1, kernel_size=1, stride=1, dilation=1, padding=0, bias=False) 
        self.fin_convpost = nn.Sequential(
            nn.BatchNorm2d(1),
            torch.nn.MaxPool2d(max_pool),
            nn.Tanh(),
            
        )
        self.im_out_size = int(64/max_pool)
        self.fc = nn.Linear(self.im_out_size*self.im_out_size, num_classes)

    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None):

        if input.dim() <5:
            i = torch.unsqueeze(input, 1)
        else:
            i = input
        b_z, ts, c, h, w = i.shape

        ii = 0
        h_next        = None
        h_next, out_next = self.rnn_model(i[:,ii], h_next)    
        
        for ii in range(1, ts):
            h_next, out_next = self.rnn_model(i[:,ii], h_next)
        
        self.fin_conv.weight.data = self.quantize.apply(self.fin_conv.weight)
        out =  self.fin_conv(out_next)
        out =  self.fin_convpost(out)
        out = out-self.thresh3   # 4, 1, 64, 64

        out = self.fc(out.view(out.size(0), -1)) 
        return out



class FULL_linear(nn.Module):
    def __init__(self, num_classes):
        super(FULL_linear, self).__init__()
        
        self.fc = nn.Linear(256*256, num_classes)

    def forward(self, input,  linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0), input_white=5):
        out = self.fc(input.view(input.size(0), -1)) 
        return out



class event_cam(nn.Module):
    def __init__(self, num_classes):
        super(event_cam, self).__init__()
        
        self.fc = nn.Linear(64*64, num_classes)
    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0)):
        
        if input.dim() <5:
            i = torch.unsqueeze(input, 1)
        else:
            i = input
        b_z, ts, c, h, w = i.shape

        ii = 0
        h_next        = None
        h_next = i[:,ii]    # set to first frame
        
        for ii in range(1, ts):
            current_frame = i[:,ii]
            out_next = torch.sign(current_frame - h_next)
            h_next = current_frame

        out = self.fc(out_next.view(out_next.size(0), -1)) 
        return out
    
class diff_cam(nn.Module):
    def __init__(self, num_classes):
        super(diff_cam, self).__init__()
        
        self.fc = nn.Linear(64*64, num_classes)
        self.num_classes  = num_classes
    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0)):
        
        if input.dim() <5:
            i = torch.unsqueeze(input, 1)
        else:
            i = input
        b_z, ts, c, h, w = i.shape

        ii = 0
        h_next        = None
        h_next = i[:,ii]    # set to first frame
        out = torch.zeros([b_z,self.num_classes]).cuda()
        for ii in range(1, ts):
            current_frame = i[:,ii]
            out_next = torch.sign(current_frame - h_next)
            h_next = current_frame

            out += self.fc(out_next.view(out_next.size(0), -1)) 
        out = out/ts
        return out

class diff_cam_2lin(nn.Module):
    def __init__(self, num_classes):
        super(diff_cam_2lin, self).__init__()
        
        self.fc = nn.Linear(64*64, 64)
        self.fcpost = nn.Sequential(
            nn.ReLU()
        )
        self.fc2 = nn.Linear(64, num_classes)
        self.num_classes  = num_classes
    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0)):
        
        if input.dim() <5:
            i = torch.unsqueeze(input, 1)
        else:
            i = input
        b_z, ts, c, h, w = i.shape

        ii = 0
        h_next        = None
        h_next = i[:,ii]    # set to first frame
        out = torch.zeros([b_z,self.num_classes]).cuda()
        for ii in range(1, ts):
            current_frame = i[:,ii]
            out_next = torch.sign(current_frame - h_next)
            h_next = current_frame
            mid =  self.fc(out_next.view(out_next.size(0), -1))
            mid = self.fcpost(mid)
            out += self.fc2(mid)
        out = out/ts
        return out

class raw_cam(nn.Module):
    def __init__(self, num_classes):
        super(raw_cam, self).__init__()
        
        self.fc = nn.Linear(256*256, num_classes)
        self.num_classes  = num_classes
    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0), input_white=None):
        
        if input.dim() <5:
            i = torch.unsqueeze(input, 1)
        else:
            i = input
        b_z, ts, c, h, w = i.shape

        ii = 0
        h_next        = None
        h_next = i[:,ii]    # set to first frame
        out = torch.zeros([b_z,self.num_classes]).cuda()
        for ii in range(1, ts):
            current_frame = i[:,ii]
            out += self.fc(current_frame.view(current_frame.size(0), -1)) 
        out = out/ts
        return out
    
class raw_cam_2lin(nn.Module):
    def __init__(self, num_classes):
        super(raw_cam_2lin, self).__init__()
        
        self.fc = nn.Linear(256*256, 64)
        self.fcpost = nn.Sequential(
            nn.ReLU()
        )
        self.fc2 = nn.Linear(64, num_classes)
        self.num_classes  = num_classes
    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0), input_white=None):
        
        if input.dim() <5:
            i = torch.unsqueeze(input, 1)
        else:
            i = input
        b_z, ts, c, h, w = i.shape

        ii = 0
        h_next        = None
        h_next = i[:,ii]    # set to first frame
        out = torch.zeros([b_z,self.num_classes]).cuda()
        for ii in range(1, ts):
            current_frame = i[:,ii]
            mid = self.fc(current_frame.view(current_frame.size(0), -1)) 
            mid = self.fcpost(mid)
            out += self.fc2(mid) 
        out = out/ts
        return out

class raw_cam_downsampled(nn.Module):
    def __init__(self, num_classes):
        super(raw_cam_downsampled, self).__init__()
        
        self.fc = nn.Linear(64*64, num_classes)
        self.num_classes  = num_classes
    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0), input_white=None):
        
        if input.dim() <5:
            i = torch.unsqueeze(input, 1)
        else:
            i = input
        b_z, ts, c, h, w = i.shape

        ii = 0
        h_next        = None
        h_next = i[:,ii]    # set to first frame
        out = torch.zeros([b_z,self.num_classes]).cuda()
        for ii in range(1, ts):
            current_frame = i[:,ii]
            out = out + self.fc(current_frame.view(current_frame.size(0), -1)) 
            # print(out.requires_grad)
        out = out/ts
        return out
    
class raw_cam_downsampled_bin_in(nn.Module):
    def __init__(self, num_classes, binarization_method):
        super(raw_cam_downsampled_bin_in, self).__init__()

        self.fc = nn.Linear(64*64, num_classes)
        self.thresh = nn.Parameter(torch.zeros(1)) # oop this is never used 
        self.num_classes  = num_classes
        self.quantize = binarization_method

    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0), input_white=None):
        
        if input.dim() <5:
            i = torch.unsqueeze(input, 1)
        else:
            i = input
        b_z, ts, c, h, w = i.shape

        ii = 0
        out = torch.zeros([b_z,self.num_classes]).cuda()
        for ii in range(1, ts):
            current_frame = i[:,ii]
            binarized = self.quantize.apply(current_frame)
            out = out + self.fc(binarized.view(binarized.size(0), -1)) 
        out = out/ts
        return out

# naive spatial temporal downsampling using the middle frame  + 1 layer linear decoder  
class nstd_1lin(nn.Module):
    def __init__(self, num_classes):
        super(nstd_1lin, self).__init__()
        
        self.fc = nn.Linear(64*64, num_classes)
        self.num_classes  = num_classes
    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0)):
        
        if input.dim() <5:
            i = torch.unsqueeze(input, 1)
        else:
            i = input
        b_z, ts, c, h, w = i.shape

        ii = int(ts/2)
        current_frame = i[:,ii]
        out = self.fc(current_frame.view(current_frame.size(0), -1)) 
        return out
    
class nstd_2lin(nn.Module):
    def __init__(self, num_classes):
        super(nstd_2lin, self).__init__()
        
        self.fc = nn.Linear(64*64, 64)
        self.fcpost = nn.Sequential(
            nn.ReLU()
        )
        self.fc2 = nn.Linear(64, num_classes)
        self.num_classes  = num_classes
    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0)):
        
        if input.dim() <5:
            i = torch.unsqueeze(input, 1)
        else:
            i = input
        b_z, ts, c, h, w = i.shape

        ii = int(ts/2)
        current_frame = i[:,ii]
        out = self.fc(current_frame.view(current_frame.size(0), -1)) 
        out = self.fcpost(out)
        out = self.fc2(out)
        return out
    
class piotr_event_cam(nn.Module): #for temporal
    def __init__(self, num_classes):
        super(piotr_event_cam, self).__init__()
        
        self.f_thresh = 0
        self.o_thresh = 0
        self.fc = nn.Linear(64*64, num_classes)


    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0)):
        if input.dim() <5:
            i = torch.unsqueeze(input, 1)
        else:
            i = input
        b_z, ts, c, h, w = i.shape

        ii = 0
        h_next        = None
        h_next = i[:,ii]    # set to first frame
        
        for ii in range(1, ts):
            current_frame = i[:,ii]
            
            forget_gate = torch.where( (current_frame- h_next)**2 > (self.f_thresh)**2, 1.0, 0.0 )
            out_ = current_frame - h_next
            out__ = torch.where(out_**2 <= (self.o_thresh)**2, 0.0, 1.0)
            sign = torch.where(out_> self.o_thresh, 1.0, -1.0)
            out_next = out__ * sign

            h_next = forget_gate * (current_frame-h_next) + h_next

        out = self.fc(out_next.view(out_next.size(0), -1)) 
        return out


class piotr_event_cam_spatial(nn.Module):
    def __init__(self, num_classes):
        super(piotr_event_cam_spatial, self).__init__()
        
        self.f_thresh = 0
        self.o_thresh = 0
        self.fc = nn.Linear(64*64, num_classes)
        self.num_classes = num_classes


    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0)):
        if input.dim() <5:
            i = torch.unsqueeze(input, 1)
        else:
            i = input
        b_z, ts, c, h, w = i.shape

        ii = 0
        h_next        = None
        h_next = i[:,ii]    # set to first frame
        
        out = torch.zeros([b_z,self.num_classes]).cuda()

        for ii in range(1, ts):
            current_frame = i[:,ii]
            
            forget_gate = torch.where( (current_frame- h_next)**2 > (self.f_thresh)**2, 1.0, 0.0 )
            out_ = current_frame - h_next
            out__ = torch.where(out_**2 <= (self.o_thresh)**2, 0.0, 1.0)
            sign = torch.where(out_> self.o_thresh, 1.0, -1.0)
            out_next = out__ * sign

            h_next = forget_gate * (current_frame-h_next) + h_next
            out += self.fc(out_next.view(out_next.size(0), -1)) 

        out = out/ts
        return out

class CAM_THAT_CNNs_ORIGINAL(nn.Module):
    def __init__(self, cnn_method, linear_method, num_classes):
        super(CAM_THAT_CNNs_ORIGINAL, self).__init__()

        self.cnn_method = cnn_method
        self.linear_method = linear_method


        self.cnn_model = nn.Conv2d(1, 16, kernel_size=4, stride=1, padding=2, bias=False)
        self.conv1post = nn.Sequential(
            nn.ReLU(),
            torch.nn.MaxPool2d(4),
            nn.Flatten())
        self.fc = nn.Linear(16*16*16, num_classes, bias = False)
        
          

    def forward(self, input):
        self.cnn_model.weight.data = self.cnn_method.apply(self.cnn_model.weight)
        self.fc.weight.data = self.linear_method.apply(self.fc.weight)
        out = self.cnn_model(input)
        out = self.conv1post(out) # 4, 16, 16, 16
        out = self.fc(out)
        return out
    


class CNN_RNN_CNN_linear(nn.Module):
    def __init__(self, cnn_model, cnn_params, rnn_model, rnn_params, final_conv_method, num_classes, max_pool=1, eight_channels=False):
        super(CNN_RNN_CNN_linear, self).__init__()

        self.cnn_method = cnn_params["cnn_method"]
        self.rnn_method = rnn_params["method"]
        self.final_cnn_method = cnn_params["final_conv_method"]
        self.quantize = final_conv_method
        self.cnn_model = cnn_model
        self.rnn_model = rnn_model
        self.thresh3 = nn.Parameter(torch.zeros(1))


        self.fin_conv = nn.Conv2d(rnn_params['rnn_hidden_size'], 1, kernel_size=1, stride=1, dilation=1, padding=0, bias=False)
        self.fin_convpost = nn.Sequential(
            nn.BatchNorm2d(1),
            torch.nn.MaxPool2d(max_pool),
            nn.Tanh(),
        )
        
        P1 = ( 1 * (1-1)- 1 + 1*(5 - 1))//2 + 1
        self.post_conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, dilation=1, padding=P1, bias=True )
        self.post_conv1_post = nn.Sequential(
            nn.ReLU(),
        )
        self.post_conv2 = nn.Conv2d(16, 1, kernel_size=5, stride=1, dilation=1, padding=P1, bias=True )
        self.post_conv2_post = nn.Sequential(
            nn.ReLU(),
        )

        self.im_out_size = int(64/max_pool)
        if eight_channels:
            self.im_out_size_other = int(64/max_pool/2)
        else:
            self.im_out_size_other = int(64/max_pool)    

        self.fc = nn.Linear(self.im_out_size*self.im_out_size_other, num_classes)

    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None):
        if (not linear_only):
            # print("GOT HERE")
            # sys.exit()
            if input.dim() <5:
                i = torch.unsqueeze(input, 1)
            else:
                i = input
            b_z, ts, c, h, w = i.shape
            ii = 0
            h_next        = None
            
            
            if print_intermediates:
                save_path = os.path.join(save_dir, str(ii))
                train_utils.ensure_dir(save_path)
            else:
                save_path = None

            
            h_next, out_next = self.rnn_model(self.cnn_model(input[:,ii], print_intermediates, save_path), h_next, print_intermediates, save_path)        # 4, 1, 64, 64
            for ii in range(1, ts):
                
                if print_intermediates:
                    save_path = os.path.join(save_dir, str(ii))
                    train_utils.ensure_dir(save_path)
                    
                h_next, out_next = self.rnn_model(self.cnn_model(input[:,ii], print_intermediates, save_path), h_next, print_intermediates, save_path)

            self.fin_conv.weight.data = self.quantize.apply(self.fin_conv.weight)
            out =  self.fin_conv(out_next)
            out =  self.fin_convpost(out)
            out = out-self.thresh3   # 4, 1, 64, 64

            out = self.post_conv1(out)
            out = self.post_conv1_post(out)
            out = self.post_conv2(out)
            out = self.post_conv2_post(out)
            out = self.fc(out.view(out.size(0), -1)) 
        else:
            self.fin_conv.weight.data = self.quantize.apply(self.fin_conv.weight)
            out =  self.fin_conv(input)
            out =  self.fin_convpost(out)
            out = out-self.thresh3   # 4, 1, 64, 64
            out = self.fc(out.view(out.size(0), -1))  

        return out
    




class CNN_RNN_linear_linear(nn.Module):
    def __init__(self, cnn_model, cnn_params, rnn_model, rnn_params, final_conv_method, num_classes, max_pool=1, eight_channels=False):
        super(CNN_RNN_linear_linear, self).__init__()

        self.cnn_method = cnn_params["cnn_method"]
        self.rnn_method = rnn_params["method"]
        self.final_cnn_method = cnn_params["final_conv_method"]
        self.quantize = final_conv_method
        self.cnn_model = cnn_model
        self.rnn_model = rnn_model
        self.thresh3 = nn.Parameter(torch.zeros(1))


        self.fin_conv = nn.Conv2d(rnn_params['rnn_hidden_size'], 1, kernel_size=1, stride=1, dilation=1, padding=0, bias=False)
        self.fin_convpost = nn.Sequential(
            nn.BatchNorm2d(1),
            torch.nn.MaxPool2d(max_pool),
            nn.Tanh(),
        )
        
        self.im_out_size = int(64/max_pool)
        if eight_channels:
            self.im_out_size_other = int(64/max_pool/2)
        else:
            self.im_out_size_other = int(64/max_pool)    


        self.fc0 = nn.Linear(self.im_out_size*self.im_out_size_other, 64)
        self.fcpost = nn.Sequential(
            nn.ReLU()
        )


        self.fc = nn.Linear(64, num_classes)

    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0)):
        if (not linear_only):
            if input.dim() <5:
                i = torch.unsqueeze(input, 1)
            else:
                i = input
            b_z, ts, c, h, w = i.shape
            ii = 0
            h_next        = None
            
            
            if print_intermediates:
                save_path = os.path.join(save_dir, str(ii))
                train_utils.ensure_dir(save_path)
            else:
                save_path = None

            
            h_next, out_next = self.rnn_model(self.cnn_model(input[:,ii], print_intermediates, save_path), h_next, print_intermediates, save_path)        # 4, 1, 64, 64
            for ii in range(1, ts):
                
                if print_intermediates:
                    save_path = os.path.join(save_dir, str(ii))
                    train_utils.ensure_dir(save_path)
                    
                h_next, out_next = self.rnn_model(self.cnn_model(input[:,ii], print_intermediates, save_path), h_next, print_intermediates, save_path)

            # can we get rid of the final conv?
            self.fin_conv.weight.data = self.quantize.apply(self.fin_conv.weight)
            out = self.fin_conv(out_next)
            out = self.fin_convpost(out)
            out = out-self.thresh3   # 4, 1, 64, 64

            out = self.fc0(out.view(out.size(0), -1))
            out = self.fcpost(out)
            out = self.fc(out) 
        else:
            self.fin_conv.weight.data = self.quantize.apply(self.fin_conv.weight)
            out =  self.fin_conv(input)
            out =  self.fin_convpost(out)
            out = out-self.thresh3   # 4, 1, 64, 64
            out = self.fc0(out.view(out.size(0), -1))  
            out = self.fcpost(out)
            out = self.fc(out) 
            
        return out
    

class cnn1_1lin(nn.Module):
    def __init__(self, cnn_model, cnn_params, final_conv_method, num_classes, max_pool=1, eight_channels=False, bfloat=False, dropout=None):
        super(cnn1_1lin, self).__init__()

        self.cnn_method = cnn_params["cnn_method"]
        self.final_cnn_method = cnn_params["final_conv_method"]
        self.quantize = final_conv_method
        self.cnn_model = cnn_model
        self.thresh3 = nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes

        self.fin_convpost = nn.Sequential(
            nn.BatchNorm2d(1),
            torch.nn.MaxPool2d(max_pool),
            nn.Tanh(),
        )
        self.im_out_size = int(64/max_pool)
        if eight_channels:
            self.im_out_size_other = int(64/max_pool/2)
        else:
            self.im_out_size_other = int(64/max_pool)

        if dropout==None:
            self.fc = nn.Linear(self.im_out_size*self.im_out_size_other, num_classes)
        else:
            self.fc = weight_drop.WeightDropLinear(self.im_out_size*self.im_out_size_other, num_classes, weight_dropout=dropout)

        self.bfloat = bfloat

    def forward(self, input, linear_only=False, print_intermediates=False, save_dir=None, min_max=(0.0,0.0), input_white=5):
        if self.bfloat:
            input = input.to(dtype=torch.bfloat16)


        if input.dim() <5:
            i = torch.unsqueeze(input, 1)
        else:
            i = input
        b_z, ts, c, h, w = i.shape
        ii = 0
        h_next        = None
        
        ii = 0
        out = torch.zeros([b_z,self.num_classes]).cuda()
        for ii in range(1, ts):
            current_frame = self.cnn_model(i[:,ii])
            out += self.fc(current_frame.view(current_frame.size(0), -1)) 
        out = out/ts
        return out
