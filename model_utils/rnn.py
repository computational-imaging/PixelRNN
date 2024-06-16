# References:
# https://github.com/bionick87/ConvGRUCell-pytorch/blob/master/Conv-GRU.py #
# convolutional lstm
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import sys
import os
import matplotlib.pyplot as plt
from training_utils import train_utils
import numpy as np


class ConvGRUCell(nn.Module):
    
    def __init__(self, params_model, rnn_method, hidden_quantization, output_bandwidth, rnn_only_flag=False):
        super(ConvGRUCell,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 64
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias']    
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size,self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
        self.Conv_ct     = nn.Conv2d(self.input_size + self.hidden_size,     self.hidden_size,self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
        dtype            = torch.FloatTensor
        self.quantize    = rnn_method
        self.rnn_method = params_model["method"]

            
    def forward(self,input,hidden):
        if hidden is None:
           size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:]) # same size as image input
           hidden    = Variable(torch.zeros(size_h)).cuda() 
        
        # quantize
        self.ConvGates.weight.data = self.quantize.apply(self.ConvGates.weight)
        self.Conv_ct.weight.data = self.quantize.apply(self.Conv_ct.weight)
        
        c1           = self.ConvGates(torch.cat((input,hidden),1)) # in: 32,2,64,64 out: 32,2,64,64
        (rt,ut)      = c1.chunk(2, 1) # 32,1,64,64, and 32,1,64,64

        reset_gate   = torch.sigmoid(rt)
        update_gate  = torch.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate,hidden)
        p1           = self.Conv_ct(torch.cat((input,gated_hidden),1)) # in: 32,2,64,64 out: 32,1,64,64

        ct           = torch.tanh(p1)                                       # candidate hidden state
        next_h       = torch.mul(update_gate,hidden) + (1-update_gate)*ct   # next hidden state (same size as input)
        return next_h, next_h        #32,1,64,64 # have next_h be the output
 

class CGRU_with_output_gate(nn.Module):
    def __init__(self,params_model, rnn_method, hidden_quantization,  output_bandwidth, rnn_only_flag=False):
        super(CGRU_with_output_gate,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 64
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias']    
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        
        self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size,self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
        self.Conv_ct     = nn.Conv2d(self.input_size + self.hidden_size,     self.hidden_size,self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
        self.Conv_out    = nn.Conv2d(self.input_size + self.hidden_size,     self.hidden_size,self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
        dtype            = torch.FloatTensor
        self.quantize    = rnn_method
        self.rnn_method = params_model["method"]


    
    def forward(self,input,hidden):
        if hidden is None:
           size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
           hidden    = Variable(torch.zeros(size_h)).cuda() 

        # print(hidden.size())
        # sys.exit()
        # quantize
        self.ConvGates.weight.data = self.quantize.apply(self.ConvGates.weight)
        self.Conv_ct.weight.data = self.quantize.apply(self.Conv_ct.weight)
        self.Conv_out.weight.data = self.quantize.apply(self.Conv_out.weight)
        


        # sys.exit()
        c1           = self.ConvGates(torch.cat((input,hidden),1))
        # sys.exit()
        (rt,ut)      = c1.chunk(2, 1)
        reset_gate   = torch.sigmoid(rt)
        update_gate  = torch.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate,hidden)
        p1           = self.Conv_ct(torch.cat((input,gated_hidden),1))
        ct           = torch.tanh(p1)
        next_h       = torch.mul(update_gate,hidden) + (1-update_gate)*ct


        output       = self.Conv_out(torch.cat((input,hidden),1))
        return next_h, output


class ConvGRU_Minimal(nn.Module):
    def __init__(self,params_model, rnn_method, hidden_quantization, output_bandwidth,  rnn_only_flag=False):
        super(ConvGRU_Minimal,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 64
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.quantize = rnn_method
        self.rnn_method = params_model["method"]

        self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size,     self.hidden_size,self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
        self.Conv_ct     = nn.Conv2d(self.input_size + self.hidden_size,     self.hidden_size,self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
        dtype            = torch.FloatTensor


    def forward(self,input,hidden, print_intermediates=False, save_dir=None, input_white=5, add_noise=0.0):

        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            hidden    = Variable(torch.zeros(size_h)).cuda() 

        self.ConvGates.weight.data = self.quantize.apply(self.ConvGates.weight)
        self.Conv_ct.weight.data = self.quantize.apply(self.Conv_ct.weight)
        ft           = self.ConvGates(torch.cat((input,hidden),1))
        forget_gate   = torch.sigmoid(ft)

        gated_hidden = torch.mul(forget_gate,hidden)
        p1           = self.Conv_ct(torch.cat((input,gated_hidden),1))
        ct           = torch.tanh(p1)
        next_h       = torch.mul(1-forget_gate,hidden) + forget_gate*ct
        return next_h, next_h


class ConvGRU_Minimal_with_output(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, output_bandwidth,  rnn_only_flag=False):
        super(ConvGRU_Minimal_with_output,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            # self.input_size  = 64 # 16 for Bose
            # # self.input_size  = 1 # 1 for so_2022_batch_norm
            self.input_size  = 64 
            # self.input_size = 16 # 16 for Bose 
            # self.input_size  = 1 # 1 for so_2022_batch_norm
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.quantize = rnn_method 
        self.rnn_method = params_model["method"]

        self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size,     self.hidden_size,self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
        self.Conv_ct     = nn.Conv2d(self.input_size + self.hidden_size,     self.hidden_size,self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
        self.Conv_out    = nn.Conv2d(self.input_size + self.hidden_size,     self.hidden_size,self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
        dtype            = torch.FloatTensor


    def forward(self,input,hidden, print_intermediates=False, save_dir=None, input_white=5, add_noise=0.0):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            hidden    = Variable(torch.zeros(size_h)).cuda() 
        # print(input.size(), hidden.size())

        self.ConvGates.weight.data = self.quantize.apply(self.ConvGates.weight)
        self.Conv_ct.weight.data = self.quantize.apply(self.Conv_ct.weight)
        self.Conv_out.weight.data = self.quantize.apply(self.Conv_out.weight)

        ft           = self.ConvGates(torch.cat((input,hidden),1))
        # ft: 4,1,64,64
        # torch.cat((input,hidden),1): 4,65,64,64
        # ConvGates.weight.data.size(): 4,65,5,5

        forget_gate   = torch.sigmoid(ft)               # 4,1,64,64
        gated_hidden = torch.mul(forget_gate,hidden)    # 4,1,64,64
        
        p1           = self.Conv_ct(torch.cat((input,gated_hidden),1))  # 4,1,64,64
        ct           = torch.tanh(p1)
        next_h       = torch.mul(1-forget_gate,hidden) + forget_gate*ct
        output       = self.Conv_out(torch.cat((input,hidden),1)) # 4,1,64,64
        return next_h, output


class CMGUO_scamp(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, output_bandwidth,  rnn_only_flag=False):
        super(CMGUO_scamp,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 64 
            self.input_size = 16 # 16 for Bose 
            self.input_size  = 1 # 1 for so_2022_batch_norm
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.quantize = rnn_method 
        self.rnn_method = params_model["method"]

        self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size,     self.hidden_size,kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.Conv_ct     = nn.Conv2d(self.input_size + self.hidden_size,     self.hidden_size,kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.Conv_out    = nn.Conv2d(self.input_size + self.hidden_size,     self.hidden_size,kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        dtype            = torch.FloatTensor


    def forward(self,input,hidden):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            hidden    = Variable(torch.zeros(size_h)).cuda() 
        # print(input.size(), hidden.size())

        self.ConvGates.weight.data = self.quantize.apply(self.ConvGates.weight)
        self.Conv_ct.weight.data = self.quantize.apply(self.Conv_ct.weight)
        self.Conv_out.weight.data = self.quantize.apply(self.Conv_out.weight)

        ft           = self.ConvGates(torch.cat((input,hidden),1))
        forget_gate  = self.quantize.apply(ft)
        gated_hidden = torch.mul(forget_gate,hidden)
        
        p1           = self.Conv_ct(torch.cat((input,gated_hidden),1))
        ct           = self.quantize.apply(p1)
        next_h       = torch.mul(1-forget_gate,hidden) + forget_gate*ct
        output       = self.Conv_out(torch.cat((input,hidden),1))
        
        return next_h, output


class rnn_1_mul(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(rnn_1_mul,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]
        self.f_init_scale = params_model["out_weight_init_scale"]

        # break up weights so hidden weights get a scaling from training
        self.ConvGates_i   = nn.Conv2d(self.input_size,   self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.Conv_out_i    = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.Conv_out_h    = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        
        self.ConvGates_h.weight.data *= self.h_init_scale
        self.Conv_out_h.weight.data  *= self.f_init_scale

    def forward(self,input,hidden, print_intermediates=False, save_dir=None, input_white=20, add_noise=0.0):
        # print(input.dtype)
        # sys.exit()
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            hidden    = torch.ones(size_h, device=torch.device('cuda'), dtype=input.dtype)
            hidden    = self.output_quantize.apply(hidden)


        self.ConvGates_i.weight.data = self.weight_quantize.apply(self.ConvGates_i.weight)
        self.ConvGates_h.weight.data = self.weight_quantize.apply(self.ConvGates_h.weight)
        self.Conv_out_i.weight.data = self.weight_quantize.apply(self.Conv_out_i.weight)
        self.Conv_out_h.weight.data = self.weight_quantize.apply(self.Conv_out_h.weight)


        fi           = self.ConvGates_i(input)
        fh           = self.ConvGates_h(hidden)    
        ft           = fi + fh 
        forget_gate  = self.gate_quantize.apply(ft)

        next_h       = forget_gate*hidden

        oi           = self.Conv_out_i(input)
        oh           = self.Conv_out_h(hidden)
        output       = oi + oh

        if print_intermediates:
            train_utils.ensure_dir(save_dir)
            hidden_x = hidden.cpu().detach().numpy()  # 1 or -1
            hidden_x = 255*(hidden_x+1)/2           # 0 255
            
            
            fi_x = fi.cpu().detach().numpy().squeeze() # -25 to 25
            fh_x = fh.cpu().detach().numpy().squeeze() # -25 to 25
            # fi_x = 255*(fi_x+25)/50         # 0 255
            # fh_x = 255*(fh_x+25)/50         # 0 255

            # Haley: 10.2023: though now we know that the white value is actually like 5. so changed from 2 to 5
            
            fi_x = fi_x*input_white +127      # 2 is the input white
            fh_x = fh_x*input_white +127      # 2 is the input white

            ft_x = ft.cpu().detach().numpy() # -50 to 50..... bruh
            # ft_x = (ft_x+50)/100 *255      # 0 255 
            ft_x = ft_x*input_white + 127      # 0 255 


            forget_gate_x = forget_gate.cpu().detach().numpy().squeeze() # -1 1
            forget_gate_x = 255*(forget_gate_x+1)/2            # 0, 255

            next_h_x = next_h.cpu().detach().numpy().squeeze()        #-1 1
            next_h_x = 255*(next_h_x+1)/2                   # 0, 255

            oi_x = oi.cpu().detach().numpy().squeeze()    
            oh_x = oh.cpu().detach().numpy().squeeze()
            # oi_x = 255*(oi_x+25)/50         # 0 255
            # oh_x = 255*(oh_x+25)/50         # 0 255
            oi_x = oi_x*input_white + 127
            oh_x = oh_x*input_white + 127

            output_x = output.cpu().detach().numpy().squeeze() # -50 50
            # output_x = 255* (output_x+50)/100
            # print(output_x.min(), output_x.max())
            # sys.exit()
            output_x = output_x*input_white +127  # the output should be -50*input_white to 50*input_white and then we add 127 to shift the whole image up for saving

            # format like in scamp:
            fi_fh_oi_oh = np.vstack([np.hstack([fi_x, fh_x ]), np.hstack([oi_x, oh_x ])])
            ft_ot = np.vstack([next_h_x,output_x])
            # print(hidden_x.max(), fi_x.max(), fh_x.max(), ft_x.max(), forget_gate_x.max(),  next_h_x.max(),oi_x.max(), oh_x.max(),  output_x.max() )
            # sys.exit()

            ConvGates_i_x = self.ConvGates_i.weight.data.cpu().detach().squeeze()
            ConvGates_h_x = self.ConvGates_h.weight.data.cpu().detach().squeeze()
            Conv_out_i_x = self.Conv_out_i.weight.data.cpu().detach().squeeze()
            Conv_out_h_x = self.Conv_out_h.weight.data.cpu().detach().squeeze()

            ConvGates_i_x = 255*(ConvGates_i_x+1)/2
            ConvGates_h_x = 255*(ConvGates_h_x+1)/2
            Conv_out_i_x = 255*(Conv_out_i_x+1)/2
            Conv_out_h_x = 255*(Conv_out_h_x+1)/2

            pad = torch.ones_like(ConvGates_i_x)*127
            vpad = torch.ones_like(torch.hstack([ConvGates_i_x, pad, ConvGates_i_x]))*127
            da_weights = torch.vstack([torch.hstack([ConvGates_i_x, pad, ConvGates_h_x]),vpad,torch.hstack([Conv_out_i_x, pad, Conv_out_h_x])])

            homef = ('/').join(save_dir.split('/')[:-3])
            
            plt.imsave(os.path.join(save_dir,"4_hidden.BMP"), hidden_x.squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"5_fi_fh_oi_oh.BMP"), fi_fh_oi_oh.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"6_ft_ot.BMP"), ft_ot.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"7_forget_gate_x.BMP"), forget_gate_x.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(homef,"rnnweights.BMP"), da_weights.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
        return next_h, output


# this adds a little noise to quantize it to binary, not ternary
class rnn_1_qt(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(rnn_1_qt,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]
        self.f_init_scale = params_model["out_weight_init_scale"]

        self.ConvGates_i   = nn.Conv2d(self.input_size,   self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.Conv_out_i    = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.Conv_out_h    = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        
        self.ConvGates_h.weight.data *= self.h_init_scale
        self.Conv_out_h.weight.data  *= self.f_init_scale

    def forward(self,input,hidden, print_intermediates=False, save_dir=None, input_white=5, add_noise=0.0, store=0):
        # print(input.dtype)
        # sys.exit()
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            hidden    = torch.ones(size_h, device=torch.device('cuda'), dtype=input.dtype)
            hidden    = self.output_quantize.apply(hidden)

        self.ConvGates_i.weight.data = self.weight_quantize.apply(self.ConvGates_i.weight)
        self.ConvGates_h.weight.data = self.weight_quantize.apply(self.ConvGates_h.weight)
        self.Conv_out_i.weight.data = self.weight_quantize.apply(self.Conv_out_i.weight)
        self.Conv_out_h.weight.data = self.weight_quantize.apply(self.Conv_out_h.weight)

        fi           = self.ConvGates_i(input)
        fh           = self.ConvGates_h(hidden)    
        ft           = fi + fh 
        forget_gate  = self.gate_quantize.apply(ft + 1e-3)

        next_h       = forget_gate*hidden
        next_h       = self.output_quantize.apply(next_h)

        oi           = self.Conv_out_i(input)
        oh           = self.Conv_out_h(hidden)
        output       = oi + oh
        # unique_values, counts = np.unique(forget_gate.cpu().detach().numpy(), return_counts=True)
        # total = np.sum(counts)
        # print("f:  ", unique_values)
        # hidden_x = hidden.cpu().detach().numpy() 
        # hidden_x = 255*(hidden_x+1)/2
        # print(np.unique(forget_gate.cpu().detach().numpy() ))
        if print_intermediates:
            train_utils.ensure_dir(save_dir)
            hidden_x = hidden.cpu().detach().numpy()  # 1 or -1
            hidden_x = 255*(hidden_x+1)/2           # 0 255
            
            
            fi_x = fi.cpu().detach().numpy().squeeze() # -25 to 25
            fh_x = fh.cpu().detach().numpy().squeeze() # -25 to 25
            # fi_x = 255*(fi_x+25)/50         # 0 255
            # fh_x = 255*(fh_x+25)/50         # 0 255

            # Haley: 10.2023: though now we know that the white value is actually like 5. so changed from 2 to 5
            
            fi_x = fi_x*input_white +127      # 2 is the input white
            fh_x = fh_x*input_white +127      # 2 is the input white

            ft_x = ft.cpu().detach().numpy() # -50 to 50..... bruh
            # ft_x = (ft_x+50)/100 *255      # 0 255 
            ft_x = ft_x*input_white + 127      # 0 255 


            forget_gate_x = forget_gate.cpu().detach().numpy().squeeze() # -1 1
            forget_gate_x = 255*(forget_gate_x+1)/2            # 0, 255

            next_h_x = next_h.cpu().detach().numpy().squeeze()        #-1 1
            next_h_x = 255*(next_h_x+1)/2                   # 0, 255

            oi_x = oi.cpu().detach().numpy().squeeze()    
            oh_x = oh.cpu().detach().numpy().squeeze()
            # oi_x = 255*(oi_x+25)/50         # 0 255
            # oh_x = 255*(oh_x+25)/50         # 0 255
            oi_x = oi_x*input_white + 127
            oh_x = oh_x*input_white + 127

            output_x = output.cpu().detach().numpy().squeeze() # -50 50
            # output_x = 255* (output_x+50)/100
            # print(output_x.min(), output_x.max())
            # sys.exit()
            output_x = output_x*input_white +127  # the output should be -50*input_white to 50*input_white and then we add 127 to shift the whole image up for saving

            # format like in scamp:
            fi_fh_oi_oh = np.vstack([np.hstack([fi_x, fh_x ]), np.hstack([oi_x, oh_x ])])
            ft_ot = np.vstack([next_h_x,output_x])
            # print(hidden_x.max(), fi_x.max(), fh_x.max(), ft_x.max(), forget_gate_x.max(),  next_h_x.max(),oi_x.max(), oh_x.max(),  output_x.max() )
            # sys.exit()

            ConvGates_i_x = self.ConvGates_i.weight.data.cpu().detach().squeeze()
            ConvGates_h_x = self.ConvGates_h.weight.data.cpu().detach().squeeze()
            Conv_out_i_x = self.Conv_out_i.weight.data.cpu().detach().squeeze()
            Conv_out_h_x = self.Conv_out_h.weight.data.cpu().detach().squeeze()

            ConvGates_i_x = 255*(ConvGates_i_x+1)/2
            ConvGates_h_x = 255*(ConvGates_h_x+1)/2
            Conv_out_i_x = 255*(Conv_out_i_x+1)/2
            Conv_out_h_x = 255*(Conv_out_h_x+1)/2

            pad = torch.ones_like(ConvGates_i_x)*127
            vpad = torch.ones_like(torch.hstack([ConvGates_i_x, pad, ConvGates_i_x]))*127
            da_weights = torch.vstack([torch.hstack([ConvGates_i_x, pad, ConvGates_h_x]),vpad,torch.hstack([Conv_out_i_x, pad, Conv_out_h_x])])

            homef = ('/').join(save_dir.split('/')[:-3])
            
            plt.imsave(os.path.join(save_dir,"4_hidden.BMP"), hidden_x.squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"5_fi_fh_oi_oh.BMP"), fi_fh_oi_oh.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"6_ft_ot.BMP"), ft_ot.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"7_forget_gate_x.BMP"), forget_gate_x.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(homef,"rnnweights.BMP"), da_weights.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
        return next_h, output


class rnn_1_mul_broken(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(rnn_1_mul_broken,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]
        self.f_init_scale = params_model["out_weight_init_scale"]

                            # nn.Conv2d(1, 16, kernel_size=k_size, stride=1, padding=P1, bias=False)
        # this works:
        # self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        # self.Conv_out    = nn.Conv2d(self.input_size + self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        
        # break up weights so hidden weights get a scaling from training
        self.ConvGates_i   = nn.Conv2d(self.input_size,   self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.Conv_out_i    = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.Conv_out_h    = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        
        self.ConvGates_h.weight.data *= self.h_init_scale
        self.Conv_out_h.weight.data  *= self.f_init_scale

        self.Conv_out_h.weight.data = self.Conv_out_i.weight.data

    def forward(self,input,hidden, print_intermediates=False, save_dir=None, input_white=5, add_noise=0.0, store=0):
        # print(input.dtype)
        # sys.exit()
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            # hidden    = Variable(torch.zeros(size_h).cuda() + input)
            # hidden    = 1e-4*torch.ones(size_h).cuda() 
            # hidden    = input
            hidden    = torch.ones(size_h, device=torch.device('cuda'), dtype=input.dtype)
            hidden    = self.output_quantize.apply(hidden)

        # this works
        # self.ConvGates.weight.data = self.weight_quantize.apply(self.ConvGates.weight)
        # self.Conv_out.weight.data = self.weight_quantize.apply(self.Conv_out.weight)
        # ft           = self.ConvGates(torch.cat((input,hidden),1))
        # forget_gate  = self.output_quantize.apply(ft)
        # next_h       = forget_gate*hidden
        # output       = self.Conv_out(torch.cat((input,hidden),1))

        self.ConvGates_i.weight.data = self.weight_quantize.apply(self.ConvGates_i.weight)
        self.ConvGates_h.weight.data = self.weight_quantize.apply(self.ConvGates_h.weight)
        self.Conv_out_i.weight.data = self.weight_quantize.apply(self.Conv_out_i.weight)
        self.Conv_out_h.weight.data = self.weight_quantize.apply(self.Conv_out_h.weight)

        fi           = self.ConvGates_i(input)
        fh           = self.ConvGates_h(hidden)
        ft           = fi + fh
        forget_gate  = self.gate_quantize.apply(ft)

        next_h       = forget_gate*hidden

        oi           = self.Conv_out_i(input)
        oh           = self.Conv_out_i(hidden)
        output       = oi + oh
        
        print("store ", store )
        if store>0:
            print(ft.size(), ft.size())
            file_path = f'/home/haleyso/rnn_ft{store}.txt'
            np.savetxt(file_path, ft.view(-1).cpu().detach().numpy())
            
            file_path = f'/home/haleyso/rnn_out{store}.txt'
            np.savetxt(file_path, output.view(-1).cpu().detach().numpy())
        

        if print_intermediates:
            train_utils.ensure_dir(save_dir)
            hidden_x = hidden.cpu().detach().numpy()  # 1 or -1
            hidden_x = 255*(hidden_x+1)/2           # 0 255
            
            
            fi_x = fi.cpu().detach().numpy().squeeze() # -25 to 25
            fh_x = fh.cpu().detach().numpy().squeeze() # -25 to 25
            # fi_x = 255*(fi_x+25)/50         # 0 255
            # fh_x = 255*(fh_x+25)/50         # 0 255

            # Haley: 10.2023: though now we know that the white value is actually like 5. so changed from 2 to 5
            
            fi_x = fi_x*input_white +127      # 2 is the input white
            fh_x = fh_x*input_white +127      # 2 is the input white

            ft_x = ft.cpu().detach().numpy() # -50 to 50..... bruh
            # ft_x = (ft_x+50)/100 *255      # 0 255 
            ft_x = ft_x*input_white + 127      # 0 255 


            forget_gate_x = forget_gate.cpu().detach().numpy().squeeze() # -1 1
            forget_gate_x = 255*(forget_gate_x+1)/2            # 0, 255

            next_h_x = next_h.cpu().detach().numpy().squeeze()        #-1 1
            next_h_x = 255*(next_h_x+1)/2                   # 0, 255

            oi_x = oi.cpu().detach().numpy().squeeze()    
            oh_x = oh.cpu().detach().numpy().squeeze()
            # oi_x = 255*(oi_x+25)/50         # 0 255
            # oh_x = 255*(oh_x+25)/50         # 0 255
            oi_x = oi_x*input_white + 127
            oh_x = oh_x*input_white + 127

            output_x = output.cpu().detach().numpy().squeeze() # -50 50
            # output_x = 255* (output_x+50)/100
            # print(output_x.min(), output_x.max())
            # sys.exit()
            output_x = output_x*input_white +127  # the output should be -50*input_white to 50*input_white and then we add 127 to shift the whole image up for saving

            # format like in scamp:
            fi_fh_oi_oh = np.vstack([np.hstack([fi_x, fh_x ]), np.hstack([oi_x, oh_x ])])
            ft_ot = np.vstack([next_h_x,output_x])
            # print(hidden_x.max(), fi_x.max(), fh_x.max(), ft_x.max(), forget_gate_x.max(),  next_h_x.max(),oi_x.max(), oh_x.max(),  output_x.max() )
            # sys.exit()

            ConvGates_i_x = self.ConvGates_i.weight.data.cpu().detach().squeeze()
            ConvGates_h_x = self.ConvGates_h.weight.data.cpu().detach().squeeze()
            Conv_out_i_x = self.Conv_out_i.weight.data.cpu().detach().squeeze()
            Conv_out_h_x = self.Conv_out_h.weight.data.cpu().detach().squeeze()

            ConvGates_i_x = 255*(ConvGates_i_x+1)/2
            ConvGates_h_x = 255*(ConvGates_h_x+1)/2
            Conv_out_i_x = 255*(Conv_out_i_x+1)/2
            Conv_out_h_x = 255*(Conv_out_h_x+1)/2

            pad = torch.ones_like(ConvGates_i_x)*127
            vpad = torch.ones_like(torch.hstack([ConvGates_i_x, pad, ConvGates_i_x]))*127
            da_weights = torch.vstack([torch.hstack([ConvGates_i_x, pad, ConvGates_h_x]),vpad,torch.hstack([Conv_out_i_x, pad, Conv_out_h_x])])

            homef = ('/').join(save_dir.split('/')[:-3])
            
            plt.imsave(os.path.join(save_dir,"4_hidden.BMP"), hidden_x.squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"5_fi_fh_oi_oh.BMP"), fi_fh_oi_oh.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"6_ft_ot.BMP"), ft_ot.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"7_forget_gate_x.BMP"), forget_gate_x.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(homef,"rnnweights.BMP"), da_weights.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
        return next_h, output


class rnn_1_mul_broken_qt(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(rnn_1_mul_broken_qt,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]
        self.f_init_scale = params_model["out_weight_init_scale"]

                            # nn.Conv2d(1, 16, kernel_size=k_size, stride=1, padding=P1, bias=False)
        # this works:
        # self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        # self.Conv_out    = nn.Conv2d(self.input_size + self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        
        # break up weights so hidden weights get a scaling from training
        self.ConvGates_i   = nn.Conv2d(self.input_size,   self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.Conv_out_i    = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.Conv_out_h    = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        
        self.ConvGates_h.weight.data *= self.h_init_scale
        self.Conv_out_h.weight.data  *= self.f_init_scale

        self.Conv_out_h.weight.data = self.Conv_out_i.weight.data



    def forward(self,input,hidden, print_intermediates=False, save_dir=None, input_white=5, add_noise=0.0):
        # print(input.dtype)
        # sys.exit()
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            # hidden    = Variable(torch.zeros(size_h).cuda() + input)
            # hidden    = 1e-4*torch.ones(size_h).cuda() 
            # hidden    = input
            hidden    = torch.ones(size_h, device=torch.device('cuda'), dtype=input.dtype)
            hidden    = self.output_quantize.apply(hidden)

        # this works
        # self.ConvGates.weight.data = self.weight_quantize.apply(self.ConvGates.weight)
        # self.Conv_out.weight.data = self.weight_quantize.apply(self.Conv_out.weight)
        # ft           = self.ConvGates(torch.cat((input,hidden),1))
        # forget_gate  = self.output_quantize.apply(ft)
        # next_h       = forget_gate*hidden
        # output       = self.Conv_out(torch.cat((input,hidden),1))

        self.ConvGates_i.weight.data = self.weight_quantize.apply(self.ConvGates_i.weight)
        self.ConvGates_h.weight.data = self.weight_quantize.apply(self.ConvGates_h.weight)
        self.Conv_out_i.weight.data = self.weight_quantize.apply(self.Conv_out_i.weight)
        self.Conv_out_h.weight.data = self.weight_quantize.apply(self.Conv_out_h.weight)

        fi           = self.ConvGates_i(input)
        fh           = self.ConvGates_h(hidden)
        ft           = fi + fh
        forget_gate  = self.gate_quantize.apply(ft+ 1e-3)

        next_h       = forget_gate*hidden

        oi           = self.Conv_out_i(input)
        oh           = self.Conv_out_i(hidden)
        output       = oi + oh
        

        if print_intermediates:
            train_utils.ensure_dir(save_dir)
            hidden_x = hidden.cpu().detach().numpy()  # 1 or -1
            hidden_x = 255*(hidden_x+1)/2           # 0 255
            
            
            fi_x = fi.cpu().detach().numpy().squeeze() # -25 to 25
            fh_x = fh.cpu().detach().numpy().squeeze() # -25 to 25
            # fi_x = 255*(fi_x+25)/50         # 0 255
            # fh_x = 255*(fh_x+25)/50         # 0 255

            # Haley: 10.2023: though now we know that the white value is actually like 5. so changed from 2 to 5
            
            fi_x = fi_x*input_white +127      # 2 is the input white
            fh_x = fh_x*input_white +127      # 2 is the input white

            ft_x = ft.cpu().detach().numpy() # -50 to 50..... bruh
            # ft_x = (ft_x+50)/100 *255      # 0 255 
            ft_x = ft_x*input_white + 127      # 0 255 


            forget_gate_x = forget_gate.cpu().detach().numpy().squeeze() # -1 1
            forget_gate_x = 255*(forget_gate_x+1)/2            # 0, 255

            next_h_x = next_h.cpu().detach().numpy().squeeze()        #-1 1
            next_h_x = 255*(next_h_x+1)/2                   # 0, 255

            oi_x = oi.cpu().detach().numpy().squeeze()    
            oh_x = oh.cpu().detach().numpy().squeeze()
            # oi_x = 255*(oi_x+25)/50         # 0 255
            # oh_x = 255*(oh_x+25)/50         # 0 255
            oi_x = oi_x*input_white + 127
            oh_x = oh_x*input_white + 127

            output_x = output.cpu().detach().numpy().squeeze() # -50 50
            # output_x = 255* (output_x+50)/100
            # print(output_x.min(), output_x.max())
            # sys.exit()
            output_x = output_x*input_white +127  # the output should be -50*input_white to 50*input_white and then we add 127 to shift the whole image up for saving

            # format like in scamp:
            fi_fh_oi_oh = np.vstack([np.hstack([fi_x, fh_x ]), np.hstack([oi_x, oh_x ])])
            ft_ot = np.vstack([next_h_x,output_x])
            # print(hidden_x.max(), fi_x.max(), fh_x.max(), ft_x.max(), forget_gate_x.max(),  next_h_x.max(),oi_x.max(), oh_x.max(),  output_x.max() )
            # sys.exit()

            ConvGates_i_x = self.ConvGates_i.weight.data.cpu().detach().squeeze()
            ConvGates_h_x = self.ConvGates_h.weight.data.cpu().detach().squeeze()
            Conv_out_i_x = self.Conv_out_i.weight.data.cpu().detach().squeeze()
            Conv_out_h_x = self.Conv_out_h.weight.data.cpu().detach().squeeze()

            ConvGates_i_x = 255*(ConvGates_i_x+1)/2
            ConvGates_h_x = 255*(ConvGates_h_x+1)/2
            Conv_out_i_x = 255*(Conv_out_i_x+1)/2
            Conv_out_h_x = 255*(Conv_out_h_x+1)/2

            pad = torch.ones_like(ConvGates_i_x)*127
            vpad = torch.ones_like(torch.hstack([ConvGates_i_x, pad, ConvGates_i_x]))*127
            da_weights = torch.vstack([torch.hstack([ConvGates_i_x, pad, ConvGates_h_x]),vpad,torch.hstack([Conv_out_i_x, pad, Conv_out_h_x])])

            homef = ('/').join(save_dir.split('/')[:-3])
            
            plt.imsave(os.path.join(save_dir,"4_hidden.BMP"), hidden_x.squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"5_fi_fh_oi_oh.BMP"), fi_fh_oi_oh.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"6_ft_ot.BMP"), ft_ot.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"7_forget_gate_x.BMP"), forget_gate_x.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(homef,"rnnweights.BMP"), da_weights.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
        return next_h, output

class rnn_2_add(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(rnn_2_add,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize  = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]

                            # nn.Conv2d(1, 16, kernel_size=k_size, stride=1, padding=P1, bias=False)
        self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        # self.Conv_ct     = nn.Conv2d(self.input_size + self.hidden_size,     self.hidden_size,kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.Conv_out    = nn.Conv2d(self.input_size + self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        dtype            = torch.FloatTensor


    def forward(self,input,hidden):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            # hidden    = Variable(torch.zeros(size_h)).cuda() 
            # hidden    = 1e-2*torch.ones(size_h).cuda() 
            hidden    = torch.ones(size_h).cuda() 
            hidden    = self.output_quantize.apply(hidden)
            # hidden    = torch.clone(input)
        # print(input.size(), hidden.size())

        # sys.exit()
        self.ConvGates.weight.data = self.weight_quantize.apply(self.ConvGates.weight)
        # self.Conv_ct.weight.data = self.quantize.apply(self.Conv_ct.weight)
        self.Conv_out.weight.data = self.weight_quantize.apply(self.Conv_out.weight)

        ft           = self.ConvGates(torch.cat((input,hidden),1))
        forget_gate  = self.gate_quantize.apply(ft)
        # print(forget_gate.max(), forget_gate.min())
        next_h = forget_gate+hidden
        
        # p1           = self.Conv_ct(torch.cat((input,gated_hidden),1))
        # ct           = self.quantize.apply(p1)
        # next_h       = torch.mul(1-forget_gate,hidden) + forget_gate*ct
        output       = self.Conv_out(torch.cat((input,hidden),1))
        
        return next_h, output


# c-SRNN
class srnn(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(srnn,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]

        self.ConvGates_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h.weight.data *= self.h_init_scale
        # self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)


    def forward(self,input,hidden , print_intermediates=False, save_dir=None):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            # hidden    = Variable(torch.zeros(size_h)).cuda()
            # hidden    = 1e-2*torch.ones(size_h).cuda() 
            hidden    = torch.ones(size_h).cuda() 
            # hidden    = input
            hidden    = self.output_quantize.apply(hidden) 

        self.ConvGates_i.weight.data = self.weight_quantize.apply(self.ConvGates_i.weight)
        self.ConvGates_h.weight.data = self.weight_quantize.apply(self.ConvGates_h.weight)
        ht           = self.ConvGates_i(input) + self.ConvGates_h(hidden)
        next_h       = self.output_quantize.apply(ht)
        
        #  hidden state, output --- for the SRNN, the output is the hidden state
        return next_h, ht

class srnno(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(srnno,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]

        self.ConvGates_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h.weight.data *= self.h_init_scale

        self.Conv_out_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.Conv_out_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        
        # self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)


    def forward(self,input,hidden , print_intermediates=False, save_dir=None):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            # hidden    = Variable(torch.zeros(size_h)).cuda()
            # hidden    = 1e-2*torch.ones(size_h).cuda() 
            hidden    = torch.ones(size_h).cuda() 
            # hidden    = input
            hidden    = self.output_quantize.apply(hidden) 

        self.ConvGates_i.weight.data = self.weight_quantize.apply(self.ConvGates_i.weight)
        self.ConvGates_h.weight.data = self.weight_quantize.apply(self.ConvGates_h.weight)
        self.Conv_out_i.weight.data = self.weight_quantize.apply(self.Conv_out_i.weight)
        self.Conv_out_h.weight.data = self.weight_quantize.apply(self.Conv_out_h.weight)
        ht           = self.ConvGates_i(input) + self.ConvGates_h(hidden)
        ot           = self.Conv_out_i(input) + self.Conv_out_h(hidden)
        next_h       = self.output_quantize.apply(ht)
        
        #  hidden state, output --- for the SRNN, the output is the hidden state
        return next_h, ot

# c-lstm
class clstm(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(clstm,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]

        self.h_scale = params_model["hidden_weight_init_scale"]
        self.i_scale = params_model["i_weight_init_scale"]
        self.f_scale = params_model["forget_weight_init_scale"]
        self.o_scale = params_model["out_weight_init_scale"]

        self.ConvGates_i_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_i_h   = nn.Conv2d( self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_f_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_f_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_o_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_o_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        
        self.ConvGates_h_h.weight.data *= self.h_scale
        self.ConvGates_i_h.weight.data *= self.i_scale
        self.ConvGates_f_h.weight.data *= self.f_scale
        self.ConvGates_o_h.weight.data *= self.o_scale

    def forward(self,input, hidden, print_intermediates=False, save_dir=None):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            # h_t    = Variable(torch.zeros(size_h)).cuda()  # 4 1 64 64 
            # c_mem_cell = Variable(torch.zeros(size_h)).cuda()

            # h_t    = 1e-4*torch.ones(size_h).cuda()
            h_t    = torch.ones(size_h).cuda() 
            # h_t    = input
            h_t    = self.output_quantize.apply(h_t) 
            
            # c_mem_cell    = 1e-4*torch.ones(size_h).cuda() 
            c_mem_cell    = torch.ones(size_h).cuda() 
            # c_mem_cell    = input
            c_mem_cell    = self.weight_quantize.apply(c_mem_cell)  # for full prec, no non-linearity is applied so just 'full'

        else:
            h_t = hidden[:,0,:,:].unsqueeze(1)
            c_mem_cell = hidden[:,1,:,:].unsqueeze(1)


        self.ConvGates_i_i.weight.data = self.weight_quantize.apply(self.ConvGates_i_i.weight)
        self.ConvGates_i_h.weight.data = self.weight_quantize.apply(self.ConvGates_i_h.weight)
        self.ConvGates_f_i.weight.data = self.weight_quantize.apply(self.ConvGates_f_i.weight)
        self.ConvGates_f_h.weight.data = self.weight_quantize.apply(self.ConvGates_f_h.weight)
        self.ConvGates_o_i.weight.data = self.weight_quantize.apply(self.ConvGates_o_i.weight)
        self.ConvGates_o_h.weight.data = self.weight_quantize.apply(self.ConvGates_o_h.weight)
        self.ConvGates_h_i.weight.data = self.weight_quantize.apply(self.ConvGates_h_i.weight)
        self.ConvGates_h_h.weight.data = self.weight_quantize.apply(self.ConvGates_h_h.weight)

        i_conv       = self.ConvGates_i_i(input) + self.ConvGates_i_h(h_t)
        i_t          = self.gate_quantize.apply(i_conv)
        f_conv       = self.ConvGates_f_i(input) + self.ConvGates_f_h(h_t)
        f_t          = self.gate_quantize.apply(f_conv)
        o_conv       = self.ConvGates_o_i(input) + self.ConvGates_o_h(h_t)
        o_t          = self.gate_quantize.apply(o_conv)


        h_hat_conv  = self.ConvGates_h_i(input) + self.ConvGates_h_h(h_t)
        h_hat       = self.output_quantize.apply(h_hat_conv)
        

        ct = f_t* c_mem_cell + i_t* h_hat
        next_h  = o_t*self.gate_quantize.apply(ct)

        memory_h_and_c = torch.cat((next_h, ct),1)
        #  memory has h and c cells, and the output is the hidden state
        return memory_h_and_c, next_h

# c-gru
class cgru(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(cgru,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]
        self.z_init_scale = params_model["z_weight_init_scale"]
        self.r_init_scale = params_model["r_weight_init_scale"]

        self.ConvGates_z_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_z_h   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_r_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_r_h  = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h_h  = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.ConvGates_z_h.weight.data *= self.z_init_scale
        self.ConvGates_r_h.weight.data *= self.r_init_scale
        self.ConvGates_h_h.weight.data *= self.h_init_scale

    def forward(self,input,hidden, print_intermediates=False, save_dir=None):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            # hidden    = Variable(torch.zeros(size_h)).cuda() 
            # hidden    = 1e-4*torch.ones(size_h).cuda() 
            hidden    = torch.ones(size_h).cuda()
            # hidden    = torch.zeros(size_h).cuda() 
            hidden    = self.output_quantize.apply(hidden) 

        self.ConvGates_z_i.weight.data = self.weight_quantize.apply(self.ConvGates_z_i.weight)
        self.ConvGates_z_h.weight.data = self.weight_quantize.apply(self.ConvGates_z_h.weight)
        self.ConvGates_r_i.weight.data = self.weight_quantize.apply(self.ConvGates_r_i.weight)
        self.ConvGates_r_h.weight.data = self.weight_quantize.apply(self.ConvGates_r_h.weight)
        self.ConvGates_h_i.weight.data = self.weight_quantize.apply(self.ConvGates_h_i.weight)
        self.ConvGates_h_h.weight.data = self.weight_quantize.apply(self.ConvGates_h_h.weight)

        r_conv      = self.ConvGates_r_i(input) + self.ConvGates_r_h(hidden)
        rt          = self.gate_quantize.apply(r_conv)
        z_conv      = self.ConvGates_z_i(input) + self.ConvGates_z_i(hidden)
        zt          = self.gate_quantize.apply(z_conv)

        h_hat_conv  = self.ConvGates_h_i(input) + self.ConvGates_h_h(rt*hidden)
        h_hat       = self.output_quantize.apply(h_hat_conv)

        # if zt.min() == -1: # if we are in the -1 1 regime, to make sure that -1 becomes 1 and 1 becomes -1,
        #     next_h      = 1/2*(1-zt)*hidden + 1/2*(1+zt)*h_hat
        # else:
        #     next_h      = (1-zt)*hidden + zt*h_hat
        next_h      = (1-zt)*hidden + zt*h_hat
        return next_h, next_h


# c-mgu
class cmgu(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(cmgu,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]
        self.f_init_scale = params_model["forget_weight_init_scale"]

        self.ConvGates_f_i   = nn.Conv2d(self.input_size ,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_f_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        
        self.ConvGates_h_i   = nn.Conv2d(self.input_size ,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        
        self.ConvGates_h_h.weight.data *= self.h_init_scale
        self.ConvGates_f_h.weight.data *= self.f_init_scale

    def forward(self,input,hidden, print_intermediates=False, save_dir=None):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            hidden    = torch.ones(size_h).cuda() 
            # hidden    = torch.zeros(size_h).cuda() 
            # hidden = input
            hidden    = self.output_quantize.apply(hidden) 

        self.ConvGates_f_i.weight.data = self.weight_quantize.apply(self.ConvGates_f_i.weight)
        self.ConvGates_f_h.weight.data = self.weight_quantize.apply(self.ConvGates_f_h.weight)
        self.ConvGates_h_i.weight.data = self.weight_quantize.apply(self.ConvGates_h_i.weight)
        self.ConvGates_h_h.weight.data = self.weight_quantize.apply(self.ConvGates_h_h.weight)


        f_conv      = self.ConvGates_f_i(input) + self.ConvGates_f_h(hidden)
        ft          = self.gate_quantize.apply(f_conv)

        h_hat_conv  = self.ConvGates_h_i(input) + self.ConvGates_h_h(ft*hidden)
        h_hat       = self.gate_quantize.apply(h_hat_conv)

        # print(f'Nans in ft {ft.isnan().any()}')
        # print(f'Nans in h_hat {h_hat.isnan().any()}')
        # if ft.min() == -1: # if we are in the -1 1 regime, to make sure that -1 becomes 1 and 1 becomes -1,
        #     next_h      = 1/2*(1-ft)*hidden + 1/2*(1+ft)*h_hat
        # else:
        #     next_h  = (1-ft)*hidden + ft*h_hat
        next_h  = (1-ft)*hidden + ft*h_hat 
        return next_h, next_h

class cmgu_v2(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(cmgu_v2,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]
        self.f_init_scale = params_model["forget_weight_init_scale"]

        self.ConvGates_f_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        
        self.ConvGates_h_h.weight.data *= self.h_init_scale
        self.ConvGates_f_h.weight.data *= self.f_init_scale


    def forward(self,input,hidden, print_intermediates=False, save_dir=None):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            hidden    = torch.ones(size_h).cuda() 
            # hidden    = torch.clone(input)
            # hidden    = 1e-5*torch.ones(size_h).cuda() 
            hidden    = self.output_quantize.apply(hidden) 
            
        self.ConvGates_f_h.weight.data = self.weight_quantize.apply(self.ConvGates_f_h.weight)
        self.ConvGates_h_i.weight.data = self.weight_quantize.apply(self.ConvGates_h_i.weight)
        self.ConvGates_h_h.weight.data = self.weight_quantize.apply(self.ConvGates_h_h.weight)

        f_conv      = self.ConvGates_f_h(hidden)
        ft          = self.gate_quantize.apply(f_conv)

        # h_hat_conv  = self.ConvGates_h(torch.cat((input, ft*hidden),1))
        h_hat_conv  = self.ConvGates_h_i(input) + self.ConvGates_h_h(ft*hidden)
        h_hat       = self.output_quantize.apply(h_hat_conv)

        next_h  = (1-ft)*hidden + ft*h_hat
        
        #  hidden state, output --- for the SRNN, the output is the hidden state
        return next_h, next_h

# fixed the gates on rnn_1_mul_noise . rnn_1_mul_noise is a subset tho so this is more general.
class rnn_1_mul_noise_fixed(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(rnn_1_mul_noise_fixed,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]
        self.f_init_scale = params_model["out_weight_init_scale"]

        self.ConvGates_i   = nn.Conv2d(self.input_size,   self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.Conv_out_i    = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.Conv_out_h    = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.rnn_threshold = nn.Parameter(torch.ones(1))

        self.ConvGates_h.weight.data *= self.h_init_scale
        self.Conv_out_h.weight.data  *= self.f_init_scale



    def forward(self,input,hidden, print_intermediates=False, save_dir=None, input_white=5, add_noise=0.0, store=0):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            hidden    = torch.ones(size_h, device=torch.device('cuda'))
            hidden    = self.output_quantize.apply(hidden)

        self.ConvGates_i.weight.data = self.weight_quantize.apply(self.ConvGates_i.weight)
        self.ConvGates_h.weight.data = self.weight_quantize.apply(self.ConvGates_h.weight)
        self.Conv_out_i.weight.data = self.weight_quantize.apply(self.Conv_out_i.weight)
        self.Conv_out_h.weight.data = self.weight_quantize.apply(self.Conv_out_h.weight)

        fi           = self.ConvGates_i(input)
        fh           = self.ConvGates_h(hidden)
        ft           = fi + fh
        ft_noise     = ft +  0.6*torch.randn_like(ft)
        forget_gate  = self.gate_quantize.apply(ft_noise-self.rnn_threshold)
    
        next_h       = forget_gate*hidden

        oi           = self.Conv_out_i(input)
        oh           = self.Conv_out_h(hidden)
        output       = oi + oh + torch.randn_like(oh)
        

        if print_intermediates:
            train_utils.ensure_dir(save_dir)
            hidden_x = hidden.cpu().detach().numpy()  # 1 or -1
            hidden_x = 255*(hidden_x+1)/2           # 0 255
            
            
            fi_x = fi.cpu().detach().numpy().squeeze() # -25 to 25
            fh_x = fh.cpu().detach().numpy().squeeze() # -25 to 25
            # fi_x = 255*(fi_x+25)/50         # 0 255
            # fh_x = 255*(fh_x+25)/50         # 0 255

            # Haley: 10.2023: though now we know that the white value is actually like 5. so changed from 2 to 5
            
            fi_x = fi_x*input_white +127      # 2 is the input white
            fh_x = fh_x*input_white +127      # 2 is the input white

            ft_x = ft.cpu().detach().numpy() # -50 to 50..... bruh
            # ft_x = (ft_x+50)/100 *255      # 0 255 
            ft_x = ft_x*input_white + 127      # 0 255 


            forget_gate_x = forget_gate.cpu().detach().numpy().squeeze() # -1 1
            forget_gate_x = 255*(forget_gate_x+1)/2            # 0, 255

            next_h_x = next_h.cpu().detach().numpy().squeeze()        #-1 1
            next_h_x = 255*(next_h_x+1)/2                   # 0, 255

            oi_x = oi.cpu().detach().numpy().squeeze()    
            oh_x = oh.cpu().detach().numpy().squeeze()
            # oi_x = 255*(oi_x+25)/50         # 0 255
            # oh_x = 255*(oh_x+25)/50         # 0 255
            oi_x = oi_x*input_white + 127
            oh_x = oh_x*input_white + 127

            output_x = output.cpu().detach().numpy().squeeze() # -50 50
            # output_x = 255* (output_x+50)/100
            # print(output_x.min(), output_x.max())
            # sys.exit()
            output_x = output_x*input_white +127  # the output should be -50*input_white to 50*input_white and then we add 127 to shift the whole image up for saving

            # format like in scamp:
            fi_fh_oi_oh = np.vstack([np.hstack([fi_x, fh_x ]), np.hstack([oi_x, oh_x ])])
            ft_ot = np.vstack([next_h_x,output_x])
            # print(hidden_x.max(), fi_x.max(), fh_x.max(), ft_x.max(), forget_gate_x.max(),  next_h_x.max(),oi_x.max(), oh_x.max(),  output_x.max() )
            # sys.exit()

            ConvGates_i_x = self.ConvGates_i.weight.data.cpu().detach().squeeze()
            ConvGates_h_x = self.ConvGates_h.weight.data.cpu().detach().squeeze()
            Conv_out_i_x = self.Conv_out_i.weight.data.cpu().detach().squeeze()
            Conv_out_h_x = self.Conv_out_h.weight.data.cpu().detach().squeeze()

            ConvGates_i_x = 255*(ConvGates_i_x+1)/2
            ConvGates_h_x = 255*(ConvGates_h_x+1)/2
            Conv_out_i_x = 255*(Conv_out_i_x+1)/2
            Conv_out_h_x = 255*(Conv_out_h_x+1)/2

            pad = torch.ones_like(ConvGates_i_x)*127
            vpad = torch.ones_like(torch.hstack([ConvGates_i_x, pad, ConvGates_i_x]))*127
            da_weights = torch.vstack([torch.hstack([ConvGates_i_x, pad, ConvGates_h_x]),vpad,torch.hstack([Conv_out_i_x, pad, Conv_out_h_x])])

            homef = ('/').join(save_dir.split('/')[:-3])
            
            plt.imsave(os.path.join(save_dir,"4_hidden.BMP"), hidden_x.squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"5_fi_fh_oi_oh.BMP"), fi_fh_oi_oh.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"6_ft_ot.BMP"), ft_ot.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"7_forget_gate_x.BMP"), forget_gate_x.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(homef,"rnnweights.BMP"), da_weights.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
        return next_h, output


class rnn_1_mul_noise(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(rnn_1_mul_noise,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]
        self.f_init_scale = params_model["out_weight_init_scale"]

        self.ConvGates_i   = nn.Conv2d(self.input_size,   self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.Conv_out_i    = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.Conv_out_h    = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.rnn_threshold = nn.Parameter(torch.ones(1))

        self.ConvGates_h.weight.data *= self.h_init_scale
        self.Conv_out_h.weight.data  *= self.f_init_scale


    def forward(self,input,hidden, print_intermediates=False, save_dir=None, input_white=5, add_noise=0.0, store=0):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            hidden    = torch.ones(size_h, device=torch.device('cuda'))
            hidden    = self.output_quantize.apply(hidden)

        self.ConvGates_i.weight.data = self.weight_quantize.apply(self.ConvGates_i.weight)
        self.ConvGates_h.weight.data = self.weight_quantize.apply(self.ConvGates_h.weight)
        self.Conv_out_i.weight.data = self.weight_quantize.apply(self.Conv_out_i.weight)
        self.Conv_out_h.weight.data = self.weight_quantize.apply(self.Conv_out_h.weight)

        fi           = self.ConvGates_i(input)
        fh           = self.ConvGates_h(hidden)
        ft           = fi + fh
        ft_noise     = ft +  0.6*torch.randn_like(ft)
        forget_gate  = self.gate_quantize.apply(ft_noise-self.rnn_threshold)
    
        next_h       = forget_gate*hidden

        oi           = self.Conv_out_i(input)
        oh           = self.Conv_out_i(hidden)
        output       = oi + oh + torch.randn_like(oh)
        

        if print_intermediates:
            train_utils.ensure_dir(save_dir)
            hidden_x = hidden.cpu().detach().numpy()  # 1 or -1
            hidden_x = 255*(hidden_x+1)/2           # 0 255
            
            
            fi_x = fi.cpu().detach().numpy().squeeze() # -25 to 25
            fh_x = fh.cpu().detach().numpy().squeeze() # -25 to 25
            # fi_x = 255*(fi_x+25)/50         # 0 255
            # fh_x = 255*(fh_x+25)/50         # 0 255

            # Haley: 10.2023: though now we know that the white value is actually like 5. so changed from 2 to 5
            
            fi_x = fi_x*input_white +127      # 2 is the input white
            fh_x = fh_x*input_white +127      # 2 is the input white

            ft_x = ft.cpu().detach().numpy() # -50 to 50..... bruh
            # ft_x = (ft_x+50)/100 *255      # 0 255 
            ft_x = ft_x*input_white + 127      # 0 255 


            forget_gate_x = forget_gate.cpu().detach().numpy().squeeze() # -1 1
            forget_gate_x = 255*(forget_gate_x+1)/2            # 0, 255

            next_h_x = next_h.cpu().detach().numpy().squeeze()        #-1 1
            next_h_x = 255*(next_h_x+1)/2                   # 0, 255

            oi_x = oi.cpu().detach().numpy().squeeze()    
            oh_x = oh.cpu().detach().numpy().squeeze()
            # oi_x = 255*(oi_x+25)/50         # 0 255
            # oh_x = 255*(oh_x+25)/50         # 0 255
            oi_x = oi_x*input_white + 127
            oh_x = oh_x*input_white + 127

            output_x = output.cpu().detach().numpy().squeeze() # -50 50
            # output_x = 255* (output_x+50)/100
            # print(output_x.min(), output_x.max())
            # sys.exit()
            output_x = output_x*input_white +127  # the output should be -50*input_white to 50*input_white and then we add 127 to shift the whole image up for saving

            # format like in scamp:
            fi_fh_oi_oh = np.vstack([np.hstack([fi_x, fh_x ]), np.hstack([oi_x, oh_x ])])
            ft_ot = np.vstack([next_h_x,output_x])
            # print(hidden_x.max(), fi_x.max(), fh_x.max(), ft_x.max(), forget_gate_x.max(),  next_h_x.max(),oi_x.max(), oh_x.max(),  output_x.max() )
            # sys.exit()

            ConvGates_i_x = self.ConvGates_i.weight.data.cpu().detach().squeeze()
            ConvGates_h_x = self.ConvGates_h.weight.data.cpu().detach().squeeze()
            Conv_out_i_x = self.Conv_out_i.weight.data.cpu().detach().squeeze()
            Conv_out_h_x = self.Conv_out_h.weight.data.cpu().detach().squeeze()

            ConvGates_i_x = 255*(ConvGates_i_x+1)/2
            ConvGates_h_x = 255*(ConvGates_h_x+1)/2
            Conv_out_i_x = 255*(Conv_out_i_x+1)/2
            Conv_out_h_x = 255*(Conv_out_h_x+1)/2

            pad = torch.ones_like(ConvGates_i_x)*127
            vpad = torch.ones_like(torch.hstack([ConvGates_i_x, pad, ConvGates_i_x]))*127
            da_weights = torch.vstack([torch.hstack([ConvGates_i_x, pad, ConvGates_h_x]),vpad,torch.hstack([Conv_out_i_x, pad, Conv_out_h_x])])

            homef = ('/').join(save_dir.split('/')[:-3])
            
            plt.imsave(os.path.join(save_dir,"4_hidden.BMP"), hidden_x.squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"5_fi_fh_oi_oh.BMP"), fi_fh_oi_oh.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"6_ft_ot.BMP"), ft_ot.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(save_dir,"7_forget_gate_x.BMP"), forget_gate_x.astype(int), vmin=0, vmax=255, cmap='gray')
            plt.imsave(os.path.join(homef,"rnnweights.BMP"), da_weights.numpy().squeeze().astype(int), vmin=0, vmax=255, cmap='gray')
        return next_h, output


class pixrnn_binarize_hidden(nn.Module):
    # the pixelrnn from supplement with interpolation, binarize the hidden state
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(pixrnn_binarize_hidden,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]
        self.f_init_scale = params_model["out_weight_init_scale"]

        self.ConvGates_f_i   = nn.Conv2d(self.input_size,   self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_f_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.ConvGates_h_i   = nn.Conv2d(self.input_size,   self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.Conv_out_i    = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.Conv_out_h    = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.rnn_f_threshold = nn.Parameter(torch.ones(1))
        self.rnn_h_threshold = nn.Parameter(torch.ones(1))

        # self.ConvGates_h.weight.data *= self.h_init_scale
        # self.Conv_out_h.weight.data  *= self.f_init_scale

    def forward(self,input,hidden, print_intermediates=False, save_dir=None):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            hidden    = torch.ones(size_h, device=torch.device('cuda'))
            # hidden    = torch.zeros(size_h, device=torch.device('cuda'))
            hidden    = self.output_quantize.apply(hidden)

        self.ConvGates_f_i.weight.data = self.weight_quantize.apply(self.ConvGates_f_i.weight)
        self.ConvGates_f_h.weight.data = self.weight_quantize.apply(self.ConvGates_f_h.weight)
        self.ConvGates_h_i.weight.data = self.weight_quantize.apply(self.ConvGates_h_i.weight)
        self.ConvGates_h_h.weight.data = self.weight_quantize.apply(self.ConvGates_h_h.weight)
        self.Conv_out_i.weight.data = self.weight_quantize.apply(self.Conv_out_i.weight)
        self.Conv_out_h.weight.data = self.weight_quantize.apply(self.Conv_out_h.weight)


        fi           = self.ConvGates_f_i(input)
        fh           = self.ConvGates_f_h(hidden)
        ft           = fi + fh
        forget_gate  = self.gate_quantize.apply(ft-self.rnn_f_threshold)
    
        hi           = self.ConvGates_h_i(input)
        hh           = self.ConvGates_h_h(hidden)
        h_tilde      = hi + hh
        h_tilde      = self.gate_quantize.apply(h_tilde-self.rnn_h_threshold)

        # if forget_gate.min() == -1: # if we are in the -1 1 regime, to make sure that -1 becomes 1 and 1 becomes -1,
        #     next_h      = 1/2*(1-forget_gate)*hidden + 1/2*(1+forget_gate)*h_tilde
        # else:
        #     next_h       = self.output_quantize.apply((1-forget_gate)*hidden + forget_gate*h_tilde)

        next_h       = self.output_quantize.apply((1-forget_gate)*hidden + forget_gate*h_tilde)
        oi           = self.Conv_out_i(input)
        oh           = self.Conv_out_i(hidden)
        output       = oi + oh 
        

        return next_h, output
    
class pixrnn(nn.Module):

    # the pixelrnn from supplement with interpolation, no binarization of the hidden state
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(pixrnn,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]
        self.f_init_scale = params_model["out_weight_init_scale"]


        self.ConvGates_f_i   = nn.Conv2d(self.input_size,   self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_f_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.ConvGates_h_i   = nn.Conv2d(self.input_size,   self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.Conv_out_i    = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.Conv_out_h    = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.rnn_f_threshold = nn.Parameter(torch.ones(1))
        self.rnn_h_threshold = nn.Parameter(torch.ones(1))

        # self.ConvGates_h.weight.data *= self.h_init_scale
        # self.Conv_out_h.weight.data  *= self.f_init_scale

    def forward(self,input,hidden, print_intermediates=False, save_dir=None):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            # hidden    = torch.zeros(size_h, device=torch.device('cuda'))
            hidden    = torch.ones(size_h, device=torch.device('cuda'))

            hidden    = self.output_quantize.apply(hidden)

        self.ConvGates_f_i.weight.data = self.weight_quantize.apply(self.ConvGates_f_i.weight)
        self.ConvGates_f_h.weight.data = self.weight_quantize.apply(self.ConvGates_f_h.weight)
        self.ConvGates_h_i.weight.data = self.weight_quantize.apply(self.ConvGates_h_i.weight)
        self.ConvGates_h_h.weight.data = self.weight_quantize.apply(self.ConvGates_h_h.weight)
        self.Conv_out_i.weight.data = self.weight_quantize.apply(self.Conv_out_i.weight)
        self.Conv_out_h.weight.data = self.weight_quantize.apply(self.Conv_out_h.weight)

        fi           = self.ConvGates_f_i(input)
        fh           = self.ConvGates_f_h(hidden)
        ft           = fi + fh
        forget_gate  = self.gate_quantize.apply(ft-self.rnn_f_threshold)
    
        hi           = self.ConvGates_h_i(input)
        hh           = self.ConvGates_h_h(hidden)
        h_tilde      = hi + hh
        h_tilde      = self.gate_quantize.apply(h_tilde-self.rnn_h_threshold)

        # if forget_gate.min() == -1: # if we are in the -1 1 regime, to make sure that -1 becomes 1 and 1 becomes -1,
        #     next_h      = 1/2*(1-forget_gate)*hidden + 1/2*(1+forget_gate)*h_tilde
        # else:
        #     next_h       = (1-forget_gate)*hidden + forget_gate*h_tilde

        next_h       = (1-forget_gate)*hidden + forget_gate*h_tilde
        oi           = self.Conv_out_i(input)
        oh           = self.Conv_out_i(hidden)
        output       = oi + oh 
        

        return next_h, output
    

class clstmo(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(clstmo, self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]

        self.h_scale = params_model["hidden_weight_init_scale"]
        self.i_scale = params_model["i_weight_init_scale"]
        self.f_scale = params_model["forget_weight_init_scale"]
        self.o_scale = params_model["out_weight_init_scale"]

        self.ConvGates_i_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_i_h   = nn.Conv2d( self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_f_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_f_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_o_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_o_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h_h   = nn.Conv2d(self.hidden_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        
        self.ConvGates_h_h.weight.data *= self.h_scale
        self.ConvGates_i_h.weight.data *= self.i_scale
        self.ConvGates_f_h.weight.data *= self.f_scale
        self.ConvGates_o_h.weight.data *= self.o_scale

    def forward(self,input, hidden, print_intermediates=False, save_dir=None, input_white=5, add_noise=0.0):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            # h_t    = Variable(torch.zeros(size_h)).cuda()  # 4 1 64 64 
            # c_mem_cell = Variable(torch.zeros(size_h)).cuda()

            # h_t    = 1e-4*torch.ones(size_h).cuda()
            h_t    = torch.ones(size_h).cuda() 
            # h_t    = input
            h_t    = self.output_quantize.apply(h_t) 
            
            # c_mem_cell    = 1e-4*torch.ones(size_h).cuda() 
            c_mem_cell    = torch.ones(size_h).cuda() 
            # c_mem_cell    = input
            c_mem_cell    = self.weight_quantize.apply(c_mem_cell)  # for full prec, no non-linearity is applied so just 'full'

        else:
            h_t = hidden[:,0,:,:].unsqueeze(1)
            c_mem_cell = hidden[:,1,:,:].unsqueeze(1)


        self.ConvGates_i_i.weight.data = self.weight_quantize.apply(self.ConvGates_i_i.weight)
        self.ConvGates_i_h.weight.data = self.weight_quantize.apply(self.ConvGates_i_h.weight)
        self.ConvGates_f_i.weight.data = self.weight_quantize.apply(self.ConvGates_f_i.weight)
        self.ConvGates_f_h.weight.data = self.weight_quantize.apply(self.ConvGates_f_h.weight)
        self.ConvGates_o_i.weight.data = self.weight_quantize.apply(self.ConvGates_o_i.weight)
        self.ConvGates_o_h.weight.data = self.weight_quantize.apply(self.ConvGates_o_h.weight)
        self.ConvGates_h_i.weight.data = self.weight_quantize.apply(self.ConvGates_h_i.weight)
        self.ConvGates_h_h.weight.data = self.weight_quantize.apply(self.ConvGates_h_h.weight)

        i_conv       = self.ConvGates_i_i(input) + self.ConvGates_i_h(h_t)
        i_t          = self.gate_quantize.apply(i_conv)
        f_conv       = self.ConvGates_f_i(input) + self.ConvGates_f_h(h_t)
        f_t          = self.gate_quantize.apply(f_conv)
        o_conv       = self.ConvGates_o_i(input) + self.ConvGates_o_h(h_t)
        o_t          = self.gate_quantize.apply(o_conv)


        h_hat_conv  = self.ConvGates_h_i(input) + self.ConvGates_h_h(h_t)
        h_hat       = self.output_quantize.apply(h_hat_conv)
        

        ct = f_t* c_mem_cell + i_t* h_hat
        next_h  = o_t*self.gate_quantize.apply(ct)

        memory_h_and_c = torch.cat((next_h, ct),1)
        #  memory has h and c cells, and the output is the hidden state
        return memory_h_and_c, o_conv


class cgruo(nn.Module):
    def __init__(self, params_model, rnn_method, hidden_quantization, gate_quantization, output_bandwidth,  rnn_only_flag=False):
        super(cgruo,self).__init__()
        if rnn_only_flag:
            self.input_size  = 1 
        else:
            self.input_size  = 1 
        self.hidden_size = params_model['rnn_hidden_size']
        self.kernel_size = params_model['kernel_size']
        self.stride = params_model['stride'] 
        self.dilation = params_model['dilation']  
        self.groups = params_model['groups']  
        self.bias = params_model['bias'] 
        self.padding = ( self.stride * (1-1)- 1 + self.dilation*(self.kernel_size - 1))//2 + 1
        self.weight_quantize = rnn_method 
        self.gate_quantize = gate_quantization
        self.output_quantize = hidden_quantization
        self.rnn_method = params_model["method"]
        self.h_init_scale = params_model["hidden_weight_init_scale"]
        self.z_init_scale = params_model["z_weight_init_scale"]
        self.r_init_scale = params_model["r_weight_init_scale"]

        self.ConvGates_z_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_z_h   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_r_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_r_h  = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h_i   = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)
        self.ConvGates_h_h  = nn.Conv2d(self.input_size,  self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=False)

        self.ConvGates_z_h.weight.data *= self.z_init_scale
        self.ConvGates_r_h.weight.data *= self.r_init_scale
        self.ConvGates_h_h.weight.data *= self.h_init_scale

    def forward(self,input,hidden, print_intermediates=False, save_dir=None):
        if hidden is None:
            size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
            # hidden    = Variable(torch.zeros(size_h)).cuda() 
            # hidden    = 1e-4*torch.ones(size_h).cuda() 
            hidden    = torch.ones(size_h).cuda()
            # hidden    = torch.zeros(size_h).cuda() 
            hidden    = self.output_quantize.apply(hidden) 

        self.ConvGates_z_i.weight.data = self.weight_quantize.apply(self.ConvGates_z_i.weight)
        self.ConvGates_z_h.weight.data = self.weight_quantize.apply(self.ConvGates_z_h.weight)
        self.ConvGates_r_i.weight.data = self.weight_quantize.apply(self.ConvGates_r_i.weight)
        self.ConvGates_r_h.weight.data = self.weight_quantize.apply(self.ConvGates_r_h.weight)
        self.ConvGates_h_i.weight.data = self.weight_quantize.apply(self.ConvGates_h_i.weight)
        self.ConvGates_h_h.weight.data = self.weight_quantize.apply(self.ConvGates_h_h.weight)

        r_conv      = self.ConvGates_r_i(input) + self.ConvGates_r_h(hidden)
        rt          = self.gate_quantize.apply(r_conv)
        z_conv      = self.ConvGates_z_i(input) + self.ConvGates_z_i(hidden)
        zt          = self.gate_quantize.apply(z_conv)

        h_hat_conv  = self.ConvGates_h_i(input) + self.ConvGates_h_h(rt*hidden)
        h_hat       = self.output_quantize.apply(h_hat_conv)

        
        # if zt.min() == -1: # if we are in the -1 1 regime, to make sure that -1 becomes 1 and 1 becomes -1,
        #     next_h      = 1/2*(1-zt)*hidden + 1/2*(1+zt)*h_hat
        # else:
        #     next_h      = (1-zt)*hidden + zt*h_hat
        next_h      = (1-zt)*hidden + zt*h_hat
        return next_h, next_h

