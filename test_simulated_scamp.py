import argparse
from data_utils import dataset_classes, utils
from training_utils import train_utils, quantization
from model_utils import get_model, cnn, rnn
import json
import numpy as np
import os
import random
import sys
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm


seed = 55
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.set_num_threads(1)

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

cnn_names = sorted(name for name in cnn.__dict__ 
                     if callable(cnn.__dict__[name]))
rnn_names = sorted(name for name in rnn.__dict__ 
                     if callable(rnn.__dict__[name]))

def test_scamp_linear(config, torch_data_path, input_white):
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
    if dataset == 'cambridge':
        num_classes = 9
    elif dataset == 'tulips1':
        num_classes = 4
    else:
        print("uh oh. no dataset chosen")
        sys.exit()


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

    
    
    torch_test_ids = []
    torch_test_labels = []
    # ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'convweights.BMP', 'rnnweights.BMP']
    classes = os.listdir(torch_data_path)
    classes.sort()
    for cls in range(len(classes)-2): # subtract 2 because now we saved the cnn and rnn weights to the folder
        class_folder = int(cls)+1
        class_path = os.path.join(torch_data_path, str(class_folder))
        videos = os.listdir(class_path)
        videos.sort()
        for vid in videos:
            vid_path = os.path.join(class_path, vid, "15", "6_ft_ot.BMP")
            torch_test_ids.append(vid_path)
            torch_test_labels.append(int(cls))

    torch_ds = dataset_classes.ScampDataset(ids= torch_test_ids, labels= torch_test_labels, input_white=input_white, pytorch_size=True)
    torch_dl = DataLoader(torch_ds, batch_size=1, num_workers=1, shuffle=False) 

    # print(torch_test_ids)
    # sys.exit()
    dl = torch_dl
    
    correct = 0
    total = len(dl)

    for xb, yb, name in tqdm(dl):
        xb = xb.cuda()
        yb = yb.cuda()
        model.eval()
        output = model(xb, linear_only=True)
        pred = output.argmax(dim=1, keepdim=True)
        corrects=pred.eq(yb.view_as(pred)).sum().item()
        correct +=corrects

        # print(pred, yb)
    print(f'Accuracy: {100*correct/total:.2f}' )
    print(f"Test Ids: {torch_test_ids}")






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate SCAMP from the print intermediates')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-sdd', '--simulated_data_folder', default='/home/haleyso/CNN_CGRU/oct2023/', type=str, help='intermediates torch path (default: {default}})')
    parser.add_argument('-iw', '--input_white', default=2, type=float, help='input white value')
    args = parser.parse_args()

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    if args.config: # load config file
        with open(args.config) as handle:
            config = json.load(handle)
    else:
        sys.exit("Add config file")

    
    test_scamp_linear(config, args.simulated_data_folder, args.input_white)

    
    




