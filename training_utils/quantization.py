from re import L
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch
import numpy as np
import math
import itertools
import sys

def get_method_func(method):
    print(method)
    if method == "full":
        method_func = full()
    elif method == "full_tanh":
        method_func = full_tanh()
    elif method == "full_sigmoid":
        method_func = full_sigmoid()
    elif method == "laurie_ternary":
        method_func = laurie_ternary()
    elif method == "laurie_insp_binary":
        method_func = laurie_insp_binary()
    elif method == "binary":
        method_func = binary()
    elif method == "suyeon_gumbel_binary":
        method_func = suyeon_gumbel_binary
    elif method == "suyeon_gumbel_ternary":
        method_func = suyeon_gumbel_ternary
    elif method == "tanh_x":
        method_func = tanh_x
    elif method == "tanh_mx":
        method_func = tanh_mx
    elif method == "sigmoid_mx":
        method_func = sigmoid_mx
    elif method == "ternary_ste":
        method_func = ternary_ste
    else:
        sys.exit("Choose method function")
    # method = method.cuda()
    return method_func


class full(Function):
    @staticmethod
    def forward(ctx, input):
        return input
    
    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output 

class full_tanh(Function):
    @staticmethod
    def apply( input):
        return torch.tanh(input)
    
    # @staticmethod 
    # def backward(ctx, grad_output):
    #     return grad_output 

class full_sigmoid(Function):
    @staticmethod
    def apply( input):
        return torch.sigmoid(input)
    
    # @staticmethod 
    # def backward(grad_output):
    #     return grad_output  

class laurie_ternary(Function):
    @staticmethod
    def forward(ctx, input):
        # print("quantizing")
        with torch.no_grad():
            noise_scale = 0.5  
            weight_mag = 1          # chooses what -1 0 1 are. so like -0.1 0 0.1 was what he originally sent
            ones = torch.ones_like(input)*weight_mag
            minusones = -ones
            r1 = torch.rand_like(input)
            r2 = -torch.rand_like(input)
            rand = torch.add(r1, r2,  alpha=1)
            #add noise and threshold weights
            noise_tensor = torch.add(input, rand,  alpha=noise_scale)

            discretized_tensor = torch.zeros_like(input)
            discretized_tensor = torch.where(noise_tensor >  1/3, ones, discretized_tensor)
            discretized_tensor = torch.where(noise_tensor < -1/3,minusones, discretized_tensor)
        return discretized_tensor

    @staticmethod # straight through estimator
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        return grad_output 

class laurie_insp_binary(Function):
    @staticmethod
    def forward(ctx, input, thresh=0, bin_type=1):
        with torch.no_grad():
            noise_scale = 0.5
            # weight_mag = 0.1
            weight_mag = 1
            ones = torch.ones_like(input)*weight_mag
            minusones = -ones
            r1 = torch.rand_like(input)
            r2 = -torch.rand_like(input)
            rand = torch.add(r1, r2,  alpha=1)
            #add noise and threshold weights
            noise_tensor = torch.add(input, rand,  alpha=noise_scale)

            if bin_type == 0: # binary type is 0 1
                discretized_tensor = torch.zeros_like(input)
                discretized_tensor = torch.where(noise_tensor > thresh, ones, discretized_tensor)
            else: # binary type is -1 1
                discretized_tensor = -1 * ones
                discretized_tensor = torch.where(noise_tensor > thresh, ones, discretized_tensor)

        return discretized_tensor
    @staticmethod # straight through estimator
    def backward(ctx, grad_output):
        return grad_output 

class binary(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        out = torch.sign(input)
        ctx.save_for_backward(input)
        return out

    @staticmethod # straight through estimator
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output

class tanh_x(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod 
    def backward(ctx, grad_output):
        input,  = ctx.saved_tensors
        m = 5
        grad_input = m*(1 - torch.pow(torch.tanh(m*input), 2))
        return grad_output*grad_input, None


class tanh_mx(Function):
    @staticmethod
    def forward(ctx, input, m=2):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        # out = torch.tanh(m*input)
        ctx.m = m
        return out

    @staticmethod 
    def backward(ctx, grad_output):
        input,  = ctx.saved_tensors
        m = ctx.m
        grad_input = m*(1 - torch.pow(torch.tanh(m*input), 2))
        return grad_output*grad_input, None


class ternary_ste(Function):
    @staticmethod
    def forward(ctx, input):
        thresh = 20
        ctx.save_for_backward(input)
        ones = torch.ones_like(input)
        minusones = -ones
        discretized_tensor = torch.zeros_like(input)
        discretized_tensor = torch.where(input >  thresh, ones, discretized_tensor)
        discretized_tensor = torch.where(input < -thresh, minusones, discretized_tensor)
        return discretized_tensor

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output
# suyeon gumbel

def score_quant_value(realWeight, s=5., qtype='binary'):
    # Here s is kinda representing the steepness
    # score of how close a value is to the quantized values
    if qtype=='binary':
        diff = torch.stack([realWeight - torch.ones_like(realWeight), realWeight + torch.ones_like(realWeight)]).permute(1,2,3,4,0).cuda()
    if qtype=='ternary':
        diff = torch.stack([realWeight - torch.ones_like(realWeight), realWeight, realWeight + torch.ones_like(realWeight)]).permute(1,2,3,4,0).cuda()
    else:
        print("didn't select a valid qtype")
        sys.exit()

    diff = diff/2 # normalize to -1 1
    z = s * diff
    # size: 16, 1, 5, 5, 2
    scores = torch.sigmoid(z) * (1 - torch.sigmoid(z)) * 4
    return scores

class suyeon_gumbel_binary(Function):

    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        tau = 0.1
        c = 300  # boost the score
        tau_max = 3.0
        scores = score_quant_value(input, (tau_max / tau)**1) * c * (tau_max / tau)**0.5
        one_hot = F.gumbel_softmax(scores, tau=tau, hard=True, dim=1)
        q_weights = (one_hot * torch.tensor([-1,1]).cuda())   # value of one hot vector  dot product with lut [-1,1]
        q_weights = q_weights.sum(dim=q_weights.ndim-1)
        return q_weights
    
    @staticmethod # straight through estimator
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        return grad_output


class suyeon_gumbel_ternary(Function):

    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        tau = 0.1
        c = 30  # boost the score
        tau_max = 3.0
        scores = score_quant_value(input, (tau_max / tau)**1, qtype='ternary') * c * (tau_max / tau)**0.5
        one_hot = F.gumbel_softmax(scores, tau=tau, hard=True, dim=1)
        q_weights = (one_hot * torch.tensor([-1,0, 1]).cuda())   # value of one hot vector  dot product with lut [-1,1]
        q_weights = q_weights.sum(dim=q_weights.ndim-1)
        return q_weights
    
    @staticmethod # straight through estimator
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        return grad_output
    
class sigmoid_mx(Function):
    @staticmethod
    def forward(ctx, input, m=6):
        ctx.save_for_backward(input)
        # out = 1/(1+torch.exp(-10*m*input))
        out = torch.where(input>0.0, 1.0 ,0.0)
        ctx.m = m
        return out

    @staticmethod 
    def backward(ctx, grad_output):
        input,  = ctx.saved_tensors
        m = ctx.m
        grad_input = (m*torch.exp(-m*input))/torch.pow((1+torch.exp(-m*input)), 2)
        return grad_output*grad_input, None