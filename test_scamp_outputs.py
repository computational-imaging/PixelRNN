import argparse
from data_utils import dataset_classes, utils
from datetime import datetime
from training_utils import train_utils, quantization
from model_utils import get_model, cnn, rnn
import json
import model_utils
import numpy as np
import os
import random
import sys
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


bit_conversion = {
    'binary':1, 
    'laurie_insp_binary': 1,
    'tanh_x':1,
    'tanh_mx':1, 
    'suyeon_gumbel_binary':1, 
    'laurie_ternary':1.5, 
    'suyeon_gumbel_ternary':1.5, 
    'full': 32,
}
    
# default size is 64. If something different, add to dict
resize_dict = { 
    "analognet1" : 256,
    "analognet2" : 28, 
    "liu_2020": 32,
    "lenet5" : 32,
    "laurie_cnn_2": 32,
    "group_so_2022": 256,
    "FULL_linear": 256,
    "event_cam": 64,
    'so_2022_new':256,
    'so_2022_batch_norm': 256,
    'CNN_bin_scamp_256':256
}

video_datasets = ['cambridge', 'hmdb51', 'jester']
image_datasets = ['mnist', 'cifar10']
paths_datasets = {
    'cambridge': "/media/data4b/haleyso/Cambridge_Hand_Gesture",
    "hmdb51": "/media/data4b/haleyso/human_motion_data/hmdb51_jpg",
    "jester": "/media/data4b/haleyso/Jester/20bn-jester-v1"
}

cnn_names = sorted(name for name in cnn.__dict__ 
                     if callable(cnn.__dict__[name]))
rnn_names = sorted(name for name in rnn.__dict__ 
                     if callable(rnn.__dict__[name]))

def test_scamp_linear(config, scamp_data_path, input_white):
    if 'seed' in config.keys():
        seed = config['seed']
        print(f'seed: {seed}')
    else:
        print("default seed is 2023")
        seed = 2023
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    cnn_model_type = config['model']['cnn_type'] 
    rnn_model_type = config['model']['rnn_type']   
    cnn_params = config['model']['cnn_params']
    rnn_params = config['model']['rnn_params']
    train_params = config['train_params']
    batch_size = config['batch_size']
    max_pool = config['max_pool']
    timesteps = config['timesteps']
    task = config["task"]
    # h, w = 224, 224
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    dataset = (config['dataset']).lower()
    num_classes = 9


    # resize shape
    if cnn_model_type in resize_dict.keys():
        resize_shape = resize_dict[cnn_model_type]
    else:
        resize_shape = 64
    print(resize_shape)
    
    # models
    # cnn model
    cnn_parameters= {
        "cnn_model_type": config['model']['cnn_type'],
        "dr_rate": cnn_params['dr_rate'],
        "kernel_size": cnn_params['kernel_size'],
        "stride": cnn_params['stride'],
        "dilation": cnn_params['dilation'],
        "groups": cnn_params['groups'],
        "bias": cnn_params['bias'],
        "cnn_method": train_params["cnn_method"],
        "final_conv_method": train_params["final_conv_method"]
    }   
    # rnn model
    rnn_parameters = {
        "rnn_model_type": config['model']['rnn_type'],
        "kernel_size": rnn_params['kernel_size'],
        "stride": rnn_params['stride'],
        "dilation": rnn_params['dilation'],
        "groups": rnn_params['groups'],
        "bias": rnn_params['bias'],
        "rnn_num_layers": rnn_params['rnn_num_layers'],
        "rnn_hidden_size": rnn_params['rnn_hidden_size'],
        "method": train_params["rnn_method"]
        
    }
    # patches... adding things to the json file
    if ("cnn_output_quantization" in train_params.keys()):
        cnn_parameters["cnn_output_quantization"] =  train_params["cnn_output_quantization"]
        # print(train_params.get("cnn_output_quantization"), type(train_params), cnn_parameters["cnn_output_quantization"])
        # sys.exit()
    else:
        cnn_parameters["cnn_output_quantization"] =  train_params["cnn_method"]
    if ("hidden_quantization" in train_params.keys()):
        rnn_parameters["hidden_quantization"] =  train_params["hidden_quantization"]
    else:
        rnn_parameters["hidden_quantization"] =  train_params["rnn_method"]
    if ("hidden_weight_init_scale" in rnn_params.keys()):
        rnn_parameters["hidden_weight_init_scale"] =  rnn_params["hidden_weight_init_scale"]
    else:
        rnn_parameters["hidden_weight_init_scale"] =  1
    if ("forget_weight_init_scale" in rnn_params.keys()):
        rnn_parameters["forget_weight_init_scale"] =  rnn_params["forget_weight_init_scale"]
    else:
        rnn_parameters["forget_weight_init_scale"] =  1
    if ("out_weight_init_scale" in rnn_params.keys()):
        rnn_parameters["out_weight_init_scale"] =  rnn_params["out_weight_init_scale"]
    else:
        rnn_parameters["out_weight_init_scale"] =  1

    if ("r_weight_init_scale" in rnn_params.keys()):
        rnn_parameters["r_weight_init_scale"] =  rnn_params["r_weight_init_scale"]
    else:
        rnn_parameters["r_weight_init_scale"] =  1

    if ("z_weight_init_scale" in rnn_params.keys()):
        rnn_parameters["z_weight_init_scale"] =  rnn_params["z_weight_init_scale"]
    else:
        rnn_parameters["z_weight_init_scale"] =  1  
    
    if ("i_weight_init_scale" in rnn_params.keys()):
        rnn_parameters["i_weight_init_scale"] =  rnn_params["i_weight_init_scale"]
    else:
        rnn_parameters["i_weight_init_scale"] =  1
    if ("gate_quantization" in train_params.keys()):
        rnn_parameters["gate_quantization"] =  train_params["gate_quantization"]
    else:
        rnn_parameters["gate_quantization"] =  "full"


    model = get_model.get_model(config, cnn_model_type, cnn_parameters, rnn_model_type, rnn_parameters, dataset, num_classes, max_pool)
    model = model.cuda()
    path2weights = config["path2weights"]
    checkpoint = torch.load(path2weights)
    model.load_state_dict(checkpoint)


    train_set_names = ['Set1_1_0001', 'Set1_1_0003', 'Set1_1_0004', 'Set1_1_0005', 'Set1_1_0006', 'Set1_1_0007', 'Set1_1_0008', 'Set1_1_0009', 'Set1_1_0010', 'Set1_1_0011', 'Set1_1_0012', 'Set1_1_0016', 'Set1_1_0017', 'Set1_1_0018', 'Set2_1_0000', 'Set2_1_0001', 'Set2_1_0002', 'Set2_1_0003', 'Set2_1_0004', 'Set2_1_0006', 'Set2_1_0007', 'Set2_1_0009', 'Set2_1_0010', 'Set2_1_0011', 'Set2_1_0012', 'Set2_1_0013', 'Set2_1_0014', 'Set2_1_0015', 'Set2_1_0016', 'Set2_1_0017', 'Set2_1_0018', 'Set2_1_0019', 'Set3_1_0000', 'Set3_1_0001', 'Set3_1_0002', 'Set3_1_0003', 'Set3_1_0004', 'Set3_1_0005', 'Set3_1_0009', 'Set3_1_0010', 'Set3_1_0011', 'Set3_1_0012', 'Set3_1_0014', 'Set3_1_0015', 'Set3_1_0016', 'Set3_1_0018', 'Set3_1_0019', 'Set4_1_0000', 'Set4_1_0001', 'Set4_1_0002', 'Set4_1_0003', 'Set4_1_0004', 'Set4_1_0005', 'Set4_1_0006', 'Set4_1_0007', 'Set4_1_0010', 'Set4_1_0011', 'Set4_1_0012', 'Set4_1_0013', 'Set4_1_0014', 'Set4_1_0015', 'Set4_1_0016', 'Set4_1_0017', 'Set4_1_0018', 'Set4_1_0019', 'Set5_1_0003', 'Set5_1_0004', 'Set5_1_0007', 'Set5_1_0009', 'Set5_1_0010', 'Set5_1_0011', 'Set5_1_0012', 'Set5_1_0013', 'Set5_1_0014', 'Set5_1_0015', 'Set5_1_0016', 'Set5_1_0017', 'Set5_1_0018', 'Set5_1_0019', 'Set1_1_0000', 'Set1_2_0001', 'Set1_2_0002', 'Set1_2_0003', 'Set1_2_0005', 'Set1_2_0007', 'Set1_2_0008', 'Set1_2_0009', 'Set1_2_0010', 'Set1_2_0011', 'Set1_2_0012', 'Set1_2_0013', 'Set1_2_0014', 'Set1_2_0015', 'Set1_2_0016', 'Set1_2_0019', 'Set2_2_0000', 'Set2_2_0001', 'Set2_2_0002', 'Set2_2_0003', 'Set2_2_0006', 'Set2_2_0007', 'Set2_2_0008', 'Set2_2_0009', 'Set2_2_0010', 'Set2_2_0011', 'Set2_2_0012', 'Set2_2_0013', 'Set2_2_0015', 'Set2_2_0016', 'Set2_2_0017', 'Set2_2_0018', 'Set2_2_0019', 'Set3_2_0000', 'Set3_2_0001', 'Set3_2_0003', 'Set3_2_0004', 'Set3_2_0006', 'Set3_2_0010', 'Set3_2_0014', 'Set3_2_0015', 'Set3_2_0016', 'Set3_2_0017', 'Set3_2_0018', 'Set3_2_0019', 'Set4_2_0000', 'Set4_2_0001', 'Set4_2_0002', 'Set4_2_0003', 'Set4_2_0004', 'Set4_2_0005', 'Set4_2_0006', 'Set4_2_0007', 'Set4_2_0008', 'Set4_2_0010', 'Set4_2_0011', 'Set4_2_0012', 'Set4_2_0013', 'Set4_2_0014', 'Set4_2_0016', 'Set4_2_0017', 'Set4_2_0018', 'Set4_2_0019', 'Set5_2_0000', 'Set5_2_0001', 'Set5_2_0002', 'Set5_2_0003', 'Set5_2_0004', 'Set5_2_0005', 'Set5_2_0006', 'Set5_2_0008', 'Set5_2_0009', 'Set5_2_0010', 'Set5_2_0012', 'Set5_2_0013', 'Set5_2_0014', 'Set5_2_0015', 'Set5_2_0016', 'Set5_2_0017', 'Set5_2_0018', 'Set1_2_0000', 'Set1_3_0001', 'Set1_3_0002', 'Set1_3_0003', 'Set1_3_0005', 'Set1_3_0006', 'Set1_3_0007', 'Set1_3_0008', 'Set1_3_0009', 'Set1_3_0010', 'Set1_3_0011', 'Set1_3_0012', 'Set1_3_0013', 'Set1_3_0014', 'Set1_3_0015', 'Set1_3_0017', 'Set1_3_0018', 'Set1_3_0019', 'Set2_3_0000', 'Set2_3_0004', 'Set2_3_0006', 'Set2_3_0007', 'Set2_3_0008', 'Set2_3_0010', 'Set2_3_0012', 'Set2_3_0013', 'Set2_3_0015', 'Set2_3_0016', 'Set2_3_0017', 'Set2_3_0018', 'Set2_3_0019', 'Set3_3_0000', 'Set3_3_0001', 'Set3_3_0002', 'Set3_3_0003', 'Set3_3_0005', 'Set3_3_0006', 'Set3_3_0008', 'Set3_3_0009', 'Set3_3_0010', 'Set3_3_0012', 'Set3_3_0013', 'Set3_3_0014', 'Set3_3_0015', 'Set3_3_0016', 'Set3_3_0018', 'Set3_3_0019', 'Set4_3_0000', 'Set4_3_0001', 'Set4_3_0002', 'Set4_3_0003', 'Set4_3_0004', 'Set4_3_0005', 'Set4_3_0006', 'Set4_3_0009', 'Set4_3_0011', 'Set4_3_0012', 'Set4_3_0013', 'Set4_3_0014', 'Set4_3_0015', 'Set4_3_0016', 'Set4_3_0017', 'Set4_3_0019', 'Set5_3_0000', 'Set5_3_0001', 'Set5_3_0002', 'Set5_3_0003', 'Set5_3_0004', 'Set5_3_0005', 'Set5_3_0006', 'Set5_3_0008', 'Set5_3_0009', 'Set5_3_0010', 'Set5_3_0011', 'Set5_3_0012', 'Set5_3_0013', 'Set5_3_0014', 'Set5_3_0017', 'Set5_3_0018', 'Set5_3_0019', 'Set1_3_0000', 'Set1_4_0001', 'Set1_4_0002', 'Set1_4_0003', 'Set1_4_0004', 'Set1_4_0005', 'Set1_4_0006', 'Set1_4_0007', 'Set1_4_0009', 'Set1_4_0010', 'Set1_4_0014', 'Set1_4_0015', 'Set1_4_0016', 'Set1_4_0017', 'Set1_4_0018', 'Set1_4_0019', 'Set2_4_0000', 'Set2_4_0002', 'Set2_4_0003', 'Set2_4_0004', 'Set2_4_0005', 'Set2_4_0006', 'Set2_4_0007', 'Set2_4_0008', 'Set2_4_0009', 'Set2_4_0010', 'Set2_4_0013', 'Set2_4_0014', 'Set2_4_0015', 'Set2_4_0016', 'Set2_4_0018', 'Set2_4_0019', 'Set3_4_0000', 'Set3_4_0001', 'Set3_4_0002', 'Set3_4_0003', 'Set3_4_0005', 'Set3_4_0006', 'Set3_4_0007', 'Set3_4_0008', 'Set3_4_0009', 'Set3_4_0010', 'Set3_4_0011', 'Set3_4_0012', 'Set3_4_0013', 'Set3_4_0014', 'Set3_4_0015', 'Set3_4_0016', 'Set3_4_0017', 'Set3_4_0018', 'Set3_4_0019', 'Set4_4_0000', 'Set4_4_0001', 'Set4_4_0003', 'Set4_4_0004', 'Set4_4_0005', 'Set4_4_0006', 'Set4_4_0007', 'Set4_4_0008', 'Set4_4_0009', 'Set4_4_0010', 'Set4_4_0011', 'Set4_4_0013', 'Set4_4_0014', 'Set4_4_0015', 'Set4_4_0016', 'Set4_4_0017', 'Set4_4_0018', 'Set4_4_0019', 'Set5_4_0000', 'Set5_4_0002', 'Set5_4_0006', 'Set5_4_0009', 'Set5_4_0010', 'Set5_4_0012', 'Set5_4_0013', 'Set5_4_0015', 'Set5_4_0016', 'Set5_4_0017', 'Set5_4_0018', 'Set1_4_0000', 'Set1_5_0001', 'Set1_5_0002', 'Set1_5_0003', 'Set1_5_0004', 'Set1_5_0005', 'Set1_5_0006', 'Set1_5_0007', 'Set1_5_0009', 'Set1_5_0011', 'Set1_5_0012', 'Set1_5_0013', 'Set1_5_0014', 'Set1_5_0015', 'Set1_5_0016', 'Set1_5_0017', 'Set1_5_0018', 'Set1_5_0019', 'Set2_5_0001', 'Set2_5_0002', 'Set2_5_0003', 'Set2_5_0004', 'Set2_5_0005', 'Set2_5_0006', 'Set2_5_0007', 'Set2_5_0008', 'Set2_5_0009', 'Set2_5_0010', 'Set2_5_0011', 'Set2_5_0012', 'Set2_5_0013', 'Set2_5_0014', 'Set2_5_0015', 'Set2_5_0016', 'Set2_5_0018', 'Set2_5_0019', 'Set3_5_0000', 'Set3_5_0002', 'Set3_5_0003', 'Set3_5_0004', 'Set3_5_0006', 'Set3_5_0008', 'Set3_5_0010', 'Set3_5_0011', 'Set3_5_0013', 'Set3_5_0014', 'Set3_5_0015', 'Set3_5_0016', 'Set3_5_0017', 'Set3_5_0018', 'Set4_5_0001', 'Set4_5_0002', 'Set4_5_0004', 'Set4_5_0005', 'Set4_5_0006', 'Set4_5_0007', 'Set4_5_0009', 'Set4_5_0012', 'Set4_5_0013', 'Set4_5_0014', 'Set4_5_0015', 'Set4_5_0016', 'Set4_5_0017', 'Set4_5_0018', 'Set4_5_0019', 'Set5_5_0001', 'Set5_5_0002', 'Set5_5_0003', 'Set5_5_0004', 'Set5_5_0005', 'Set5_5_0006', 'Set5_5_0007', 'Set5_5_0010', 'Set5_5_0011', 'Set5_5_0013', 'Set5_5_0014', 'Set5_5_0015', 'Set5_5_0017', 'Set5_5_0018', 'Set5_5_0019', 'Set1_5_0000', 'Set1_6_0002', 'Set1_6_0003', 'Set1_6_0004', 'Set1_6_0005', 'Set1_6_0006', 'Set1_6_0007', 'Set1_6_0008', 'Set1_6_0010', 'Set1_6_0012', 'Set1_6_0013', 'Set1_6_0014', 'Set1_6_0016', 'Set1_6_0017', 'Set1_6_0018', 'Set1_6_0019', 'Set2_6_0001', 'Set2_6_0002', 'Set2_6_0003', 'Set2_6_0004', 'Set2_6_0005', 'Set2_6_0006', 'Set2_6_0007', 'Set2_6_0009', 'Set2_6_0010', 'Set2_6_0011', 'Set2_6_0012', 'Set2_6_0013', 'Set2_6_0014', 'Set2_6_0015', 'Set2_6_0017', 'Set2_6_0018', 'Set2_6_0019', 'Set3_6_0000', 'Set3_6_0001', 'Set3_6_0002', 'Set3_6_0003', 'Set3_6_0004', 'Set3_6_0005', 'Set3_6_0006', 'Set3_6_0007', 'Set3_6_0008', 'Set3_6_0009', 'Set3_6_0011', 'Set3_6_0012', 'Set3_6_0014', 'Set3_6_0015', 'Set3_6_0016', 'Set3_6_0017', 'Set3_6_0018', 'Set3_6_0019', 'Set4_6_0000', 'Set4_6_0001', 'Set4_6_0002', 'Set4_6_0003', 'Set4_6_0005', 'Set4_6_0007', 'Set4_6_0008', 'Set4_6_0009', 'Set4_6_0011', 'Set4_6_0012', 'Set4_6_0013', 'Set4_6_0014', 'Set4_6_0015', 'Set4_6_0016', 'Set4_6_0017', 'Set4_6_0018', 'Set4_6_0019', 'Set5_6_0000', 'Set5_6_0001', 'Set5_6_0002', 'Set5_6_0004', 'Set5_6_0009', 'Set5_6_0012', 'Set5_6_0013', 'Set5_6_0014', 'Set5_6_0015', 'Set5_6_0016', 'Set5_6_0018', 'Set5_6_0019', 'Set1_6_0001', 'Set1_7_0001', 'Set1_7_0002', 'Set1_7_0003', 'Set1_7_0005', 'Set1_7_0006', 'Set1_7_0007', 'Set1_7_0008', 'Set1_7_0009', 'Set1_7_0010', 'Set1_7_0011', 'Set1_7_0012', 'Set1_7_0013', 'Set1_7_0014', 'Set1_7_0018', 'Set1_7_0019', 'Set2_7_0001', 'Set2_7_0003', 'Set2_7_0004', 'Set2_7_0005', 'Set2_7_0007', 'Set2_7_0008', 'Set2_7_0009', 'Set2_7_0010', 'Set2_7_0011', 'Set2_7_0012', 'Set2_7_0013', 'Set2_7_0014', 'Set2_7_0015', 'Set2_7_0016', 'Set2_7_0017', 'Set2_7_0019', 'Set3_7_0001', 'Set3_7_0002', 'Set3_7_0003', 'Set3_7_0004', 'Set3_7_0005', 'Set3_7_0008', 'Set3_7_0010', 'Set3_7_0011', 'Set3_7_0012', 'Set3_7_0013', 'Set3_7_0015', 'Set3_7_0018', 'Set3_7_0019', 'Set4_7_0000', 'Set4_7_0001', 'Set4_7_0002', 'Set4_7_0003', 'Set4_7_0004', 'Set4_7_0005', 'Set4_7_0006', 'Set4_7_0007', 'Set4_7_0008', 'Set4_7_0009', 'Set4_7_0010', 'Set4_7_0011', 'Set4_7_0013', 'Set4_7_0014', 'Set4_7_0015', 'Set4_7_0016', 'Set4_7_0017', 'Set4_7_0018', 'Set4_7_0019', 'Set5_7_0000', 'Set5_7_0003', 'Set5_7_0004', 'Set5_7_0005', 'Set5_7_0006', 'Set5_7_0007', 'Set5_7_0008', 'Set5_7_0009', 'Set5_7_0010', 'Set5_7_0012', 'Set5_7_0013', 'Set5_7_0015', 'Set5_7_0016', 'Set5_7_0017', 'Set5_7_0018', 'Set5_7_0019', 'Set1_7_0000', 'Set1_8_0002', 'Set1_8_0004', 'Set1_8_0005', 'Set1_8_0006', 'Set1_8_0007', 'Set1_8_0008', 'Set1_8_0011', 'Set1_8_0013', 'Set1_8_0014', 'Set1_8_0015', 'Set1_8_0016', 'Set1_8_0018', 'Set1_8_0019', 'Set2_8_0000', 'Set2_8_0001', 'Set2_8_0003', 'Set2_8_0004', 'Set2_8_0005', 'Set2_8_0006', 'Set2_8_0007', 'Set2_8_0008', 'Set2_8_0009', 'Set2_8_0010', 'Set2_8_0011', 'Set2_8_0012', 'Set2_8_0013', 'Set2_8_0014', 'Set2_8_0015', 'Set2_8_0016', 'Set2_8_0017', 'Set2_8_0018', 'Set2_8_0019', 'Set3_8_0000', 'Set3_8_0004', 'Set3_8_0005', 'Set3_8_0006', 'Set3_8_0007', 'Set3_8_0008', 'Set3_8_0010', 'Set3_8_0011', 'Set3_8_0012', 'Set3_8_0014', 'Set3_8_0015', 'Set3_8_0016', 'Set3_8_0018', 'Set3_8_0019', 'Set4_8_0000', 'Set4_8_0001', 'Set4_8_0002', 'Set4_8_0003', 'Set4_8_0005', 'Set4_8_0007', 'Set4_8_0008', 'Set4_8_0009', 'Set4_8_0010', 'Set4_8_0011', 'Set4_8_0012', 'Set4_8_0013', 'Set4_8_0014', 'Set4_8_0015', 'Set4_8_0016', 'Set4_8_0017', 'Set4_8_0018', 'Set5_8_0001', 'Set5_8_0002', 'Set5_8_0003', 'Set5_8_0004', 'Set5_8_0005', 'Set5_8_0007', 'Set5_8_0008', 'Set5_8_0009', 'Set5_8_0010', 'Set5_8_0011', 'Set5_8_0012', 'Set5_8_0014', 'Set5_8_0015', 'Set5_8_0016', 'Set5_8_0017', 'Set5_8_0019', 'Set1_8_0000', 'Set1_9_0001', 'Set1_9_0003', 'Set1_9_0004', 'Set1_9_0006', 'Set1_9_0007', 'Set1_9_0009', 'Set1_9_0010', 'Set1_9_0011', 'Set1_9_0012', 'Set1_9_0013', 'Set1_9_0014', 'Set1_9_0015', 'Set1_9_0016', 'Set1_9_0017', 'Set1_9_0018', 'Set1_9_0019', 'Set2_9_0000', 'Set2_9_0001', 'Set2_9_0002', 'Set2_9_0003', 'Set2_9_0004', 'Set2_9_0005', 'Set2_9_0006', 'Set2_9_0010', 'Set2_9_0011', 'Set2_9_0012', 'Set2_9_0013', 'Set2_9_0016', 'Set2_9_0017', 'Set2_9_0018', 'Set2_9_0019', 'Set3_9_0001', 'Set3_9_0002', 'Set3_9_0003', 'Set3_9_0005', 'Set3_9_0006', 'Set3_9_0007', 'Set3_9_0010', 'Set3_9_0011', 'Set3_9_0012', 'Set3_9_0013', 'Set3_9_0014', 'Set3_9_0015', 'Set3_9_0017', 'Set3_9_0018', 'Set4_9_0000', 'Set4_9_0001', 'Set4_9_0002', 'Set4_9_0003', 'Set4_9_0004', 'Set4_9_0005', 'Set4_9_0007', 'Set4_9_0008', 'Set4_9_0009', 'Set4_9_0012', 'Set4_9_0013', 'Set4_9_0014', 'Set4_9_0015', 'Set4_9_0016', 'Set4_9_0017', 'Set4_9_0018', 'Set4_9_0019', 'Set5_9_0000', 'Set5_9_0001', 'Set5_9_0002', 'Set5_9_0003', 'Set5_9_0004', 'Set5_9_0007', 'Set5_9_0008', 'Set5_9_0009', 'Set5_9_0010', 'Set5_9_0011', 'Set5_9_0012', 'Set5_9_0013', 'Set5_9_0014', 'Set5_9_0016', 'Set5_9_0017', 'Set5_9_0018', 'Set5_9_0019', 'Set1_9_0000']
    
    train_ids = []
    train_labels = []
    test_ids = []
    test_labels = []
    
    classes = os.listdir(scamp_data_path)
    classes.sort()
    for cls in range(len(classes)): 
        class_path = os.path.join(scamp_data_path, str(cls))
        videos = os.listdir(class_path)
        videos.sort()
        for vid in videos:
            vid_path = os.path.join(class_path, vid, "15_output.BMP")
            if os.path.exists(vid_path):
                if vid in train_set_names:
                    train_ids.append(vid_path)
                    train_labels.append(int(cls))
                else:
                    test_ids.append(vid_path)
                    test_labels.append(int(cls))

    scamp_train_ds = dataset_classes.ScampDataset(ids= train_ids, labels=train_labels, input_white=input_white)
    scamp_test_ds = dataset_classes.ScampDataset(ids= test_ids, labels=test_labels, input_white=input_white)
    scamp_train_dl = DataLoader(scamp_train_ds, batch_size=1, num_workers=1, shuffle=False) 
    scamp_test_dl = DataLoader(scamp_test_ds, batch_size=1, num_workers=1, shuffle=False) 
    

    # run train set
    correct = 0
    total = len(scamp_train_dl)
    if total !=0:
        for xb, yb, name in tqdm(scamp_train_dl):
            # print(xb.min(), xb.max(), xb.mean())
            xb = xb.cuda()
            yb = yb.cuda()
            model.eval()
            output = model(xb, linear_only=True)
            pred = output.argmax(dim=1, keepdim=True)
            corrects=pred.eq(yb.view_as(pred)).sum().item()
            correct +=corrects
        print(f'Train Accuracy: {100*correct/total:.2f}' )
    else:
        print("No train videos in this folder.")

    # run test set
    correct = 0
    total = len(scamp_test_dl)
    if total !=0:
        for xb, yb, name in tqdm(scamp_test_dl):
            xb = xb.cuda()
            yb = yb.cuda()
            model.eval()
            output = model(xb, linear_only=True)
            pred = output.argmax(dim=1, keepdim=True)
            corrects=pred.eq(yb.view_as(pred)).sum().item()
            correct +=corrects
        print(f'Test Accuracy: {100*correct/total:.2f}' )
    else:
        print("No test videos in this folder.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNN CNN- scamp linear layer tester')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-sd', '--scamp_data', default='/media/data4b/haleyso/cambridge_outputs/', type=str, help='scamp file path (default: {default})')
    parser.add_argument('-iw', '--input_white', default=10, type=float, help='input white in scamp (default: {default})')
    args = parser.parse_args()

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    if args.config: # load config file
        with open(args.config) as handle:
            config = json.load(handle)
    else:
        sys.exit("Add config file")


    test_scamp_linear(config, args.scamp_data, args.input_white)

    
    




