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
from prettytable import PrettyTable

seed = 55
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.set_num_threads(1)

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

def get_weights(config, scamp=False):

    if 'seed' in config.keys():
        seed = config['seed']
        print(f'seed: {seed}')
    else:
        print("default seed is 2023")
        seed = 2023
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    # dataset 
    dataset = (config['dataset']).lower()
    all_vids, all_labels, cats, csv_file = utils.get_videos(dataset)
    # print("Number of videos: ", len(all_vids), " | Number of categories: ", len(cats))

    # number of classes we're using
    if config["train_params"]["num_classes"] != "all":
        num_classes = config["train_params"]["num_classes"]
    else:
        num_classes = len(cats)

    # labels2number dictionary
    labels2number = {}
    cats.sort()
    for ind, uc in enumerate(cats):
        labels2number[uc] = ind 
    
   
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


    model = get_model.get_model(config,cnn_model_type, cnn_parameters, rnn_model_type, rnn_parameters, dataset, num_classes, max_pool)
    path2weights = config["path2weights"]
    checkpoint = torch.load(path2weights)
    model.load_state_dict(checkpoint)


    f = open("/home/haleyso/PixelRNN/lip_iw10.txt", "w")

    
    multiplier = 100
    conv_weight_custom = ['cnn_model.conv1.weight','cnn_model.conv2.weight', ]
    for name, param in model.named_parameters():
        if param.requires_grad:
            
            # make format for SCAMP
            if (scamp):
                print("Formatting for scamp")
                f.write(name.lower()+ ":\n")
                if "conv" in name.lower() and "fin" not in name.lower():
                    list_of_weights = []
                    for kern in range(param.data.shape[0]):
                        scamp_weights = []
                        da_weight = param.data[kern,0,:,:]
                        # da_weight = torch.flip(da_weight, [1])

                        scamp_weights.append(da_weight[4,:])
                        scamp_weights.append(da_weight[3,:])
                        scamp_weights.append(da_weight[2,:])
                        scamp_weights.append(da_weight[1,:])
                        scamp_weights.append(da_weight[0,:])

                        
                        scamp_weights = torch.hstack(scamp_weights)
                        print(scamp_weights)
                        # sys.exit()
                        scamp_weights = scamp_weights.to(torch.int).numpy()
                        scamp_weights = np.char.mod('%d', scamp_weights)

                        
                        scamp_weights = "        {" + ",".join(scamp_weights)+ "},\n"

                        list_of_weights.append(scamp_weights)
                    if 'Conv' not in name:
                        # then reorder cnn weight order for scamp
                        reorder_to = [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16 ] # written in scamp quadrants
                        reorder = []
                        for i in reorder_to:
                            reorder.append(list_of_weights[i-1])

                        list_of_weights = reorder

                    print(len(list_of_weights))
                    # sys.exit()
                    for i in range(len(list_of_weights)):
                        f.write("    { \n" )
                        f.write(list_of_weights[i])
                        f.write("    }, \n" )

                # elif "fc.w" in name.lower():
                #     print(name.lower(), param.data.shape)
                #     print(param.data * multiplier)
                else:
                    f.write( str(param.data.tolist())+ "\n" )

            else:
                f.write(name + ": " +  str(param.data.tolist())+ "\n" )
                # print(param.data.shape, type(param.data))
    f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNN CNN')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-sc', '--scamp', default=False, type=bool, help='Are you extracting weights to be read into scamp?')
    
    args = parser.parse_args()

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    if args.config: # load config file
        with open(args.config) as handle:
            config = json.load(handle)
    else:
        print("add the config file")
        sys.exit()
    

    get_weights(config, args.scamp)

    