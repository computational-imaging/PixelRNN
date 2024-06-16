''' This folder tries to streamline the training. 
    - select dataset and size
    - select cnn feature extraction model parameters
    - select rnn feature extraction model parameters
    - select training parameters
    - select binary or ternary quantization in forward pass
    - select kind of backward pass model estimator 
'''

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
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

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
    'sigmoid_mx':1,
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
    "raw_cam": 256,
    "event_cam": 64,
    'so_2022_new':256,
    'so_2022_batch_norm': 256,
    'CNN_bin_scamp_256':256,
    'Minotaur':256,
    'raw_cam_downsampled':64
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
# loss_choices = sorted(name for name in losses.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(losses.__dict__[name]))

# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def count_parameters(model, cnn_method, rnn_method, final_conv_method, linear_method):
    table = PrettyTable(["Modules", "Parameters", "Quantization","Bits"])
    total_params = 0
    total_in_bits = 0
    extractor_params = 0
    extractor_in_bits = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        if 'cnn_model' in name:
            bits = bit_conversion[cnn_method] * params
            quant = bit_conversion[cnn_method]
            extractor_params+=params
            extractor_in_bits+=bits
        elif 'rnn_model' in name:
            bits = bit_conversion[rnn_method] * params
            quant = bit_conversion[rnn_method]
            extractor_params+=params
            extractor_in_bits+=bits
        elif 'fin_conv' in name:
            bits = bit_conversion[final_conv_method] * params 
            quant = bit_conversion[final_conv_method]
            extractor_params+=params
            extractor_in_bits+=bits
        elif 'fc' in name:
            bits = bit_conversion[linear_method] * params 
            quant = bit_conversion[linear_method]
        elif name=='thresh3':
            bits = 8 * params 
            quant = 8
            extractor_params+=params
            extractor_in_bits+=bits
        else:
            break
        table.add_row([name, params, quant, bits])
        total_params+=params
        total_in_bits+=bits
    print(table)
    print(f"Total Trainable Params: {total_params}, Extractor Params {extractor_params}")
    print(f"Total bits: {total_in_bits}, Total megabytes: {total_in_bits/(8*1000000)}")
    print(f"Extractor bits: {extractor_in_bits}, Extractor megabytes: {extractor_in_bits/(8*1000000)}")
    return total_params


def train_videos(config, resume):
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
    if "seed" in config.keys():
        seed = config["seed"]
    else:
        seed = 2023

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    name = config['name']+ str(seed)

    # dataset 
    dataset = (config['dataset']).lower()    
    all_vids, all_labels, cats, csv_file = utils.get_videos(dataset)
    print("Number of videos: ", len(all_vids), " | Number of categories: ", len(cats))

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
    
    # split data
    print("Number of classes we're using ", num_classes)
    train_ids, train_labels, val_ids, val_labels, test_ids, test_labels = utils.split_data(all_vids, all_labels, num_classes, labels2number, csv_file=csv_file, val_size=0.1,test_size=0.1)
    print("Number of training: ", len(train_ids), " | Number of validation: ", len(val_ids), " | Number of test:", len(test_ids))

    
    # resize shape
    if cnn_model_type in resize_dict.keys():
        resize_shape = resize_dict[cnn_model_type]
    else:
        resize_shape = 64
    
    # Image Transforms
    if dataset == 'egogesture':
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(240),
            transforms.Normalize(mean, std),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([resize_shape,resize_shape], interpolation=transforms.InterpolationMode.BILINEAR),
            ]) 
    elif dataset =='cambridge' and config['model']['cnn_type'] !='CNN_k5_single_layer_relu_mp_thresh_quantized' :
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([resize_shape,resize_shape], interpolation=transforms.InterpolationMode.BILINEAR),
            ])  
    elif dataset =='tulips1':
        train_transformer = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1)), 
            transforms.ToTensor(),
            # transforms.ColorJitter(0.5,0.5,0.5,0),
            # train_utils.AddGaussianNoise(0., 1.),
            transforms.Normalize(mean, std),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([resize_shape,resize_shape], interpolation=transforms.InterpolationMode.BILINEAR),
            ]) 
    else:
        train_transformer = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),   
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([resize_shape,resize_shape], interpolation=transforms.InterpolationMode.BILINEAR),
            ]) 

    test_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([resize_shape,resize_shape], interpolation=transforms.InterpolationMode.BILINEAR),
        ])
    
    if ("duplicate_first_frame" in train_params.keys()):
        use_first = train_params["duplicate_first_frame"]
    else:
        use_first = False

    # dataset
    train_ds = dataset_classes.VideoDataset(ids= train_ids, labels= train_labels, transform= train_transformer, timesteps=timesteps, labels2number=labels2number, val=False, use_first=use_first )
    if config['dataset'] == "tulips1":
        val_ds = dataset_classes.VideoDataset(ids= val_ids, labels= val_labels, transform= test_transformer, timesteps=timesteps, labels2number=labels2number,val=True, tulips=True, use_first=use_first)
        test_ds = dataset_classes.VideoDataset(ids= test_ids, labels= test_labels, transform= test_transformer, timesteps=timesteps, labels2number=labels2number,val=True, tulips=True, use_first=use_first)
    else: 
        val_ds = dataset_classes.VideoDataset(ids= val_ids, labels= val_labels, transform= test_transformer, timesteps=timesteps, labels2number=labels2number,val=True)
        test_ds = dataset_classes.VideoDataset(ids= test_ids, labels= test_labels, transform= test_transformer, timesteps=timesteps, labels2number=labels2number,val=True)

    # data_loaders
    train_dl = DataLoader(train_ds, batch_size= batch_size, num_workers=4, shuffle=True, collate_fn= collate_fn_rnn) # does shuffle shuffle sequences??
    val_dl = DataLoader(val_ds, batch_size= batch_size, num_workers=4, shuffle=False, collate_fn= collate_fn_rnn)  
    test_dl = DataLoader(test_ds, batch_size= batch_size, num_workers=4, shuffle=False, collate_fn= collate_fn_rnn)  

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

    # patches... adding things to the json file...
    if ("cnn_output_quantization" in train_params.keys()):
        cnn_parameters["cnn_output_quantization"] =  train_params["cnn_output_quantization"]
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
    
    if ("bit_precision" in train_params.keys()):
        if train_params["bit_precision"] == 'bfloat16':
            bfloat16 = True
    else:
        bfloat16 = False

    if ("dropout" in train_params.keys()):
        dropout = train_params["dropout"]
    else:
        dropout = 0.0

    if bfloat16:
        model = get_model.get_model(config, cnn_model_type, cnn_parameters, rnn_model_type, rnn_parameters, dataset, num_classes, max_pool, bfloat16=True)
    else:
        model = get_model.get_model(config, cnn_model_type, cnn_parameters, rnn_model_type, rnn_parameters, dataset, num_classes, max_pool, dropout=dropout)

    model = model.cuda()


    # print model size https://discuss.pytorch.org/t/finding-model-size/130275     
    count_parameters(model, config['train_params']['cnn_method'], config['train_params']['rnn_method'], config['train_params']['final_conv_method'], config['train_params']['linear_method'])
    path2weights = config["path2weights"].split('.')[0]+"_seed_" + str(seed)+ ".pt"
    save_dir = os.path.join(config["save_dir"], dataset)
    log_dir = os.path.join(config["log_dir"], dataset)
    save_folder_name = os.path.join(save_dir, config["name"] )
    log_folder_name = os.path.join(log_dir, config["name"] )

    os.makedirs(save_folder_name, exist_ok=True)
    os.makedirs(os.path.join( log_folder_name ), exist_ok=True)
    if resume:
        print("resuming training")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
        model=model.cuda()
    # else:
    torch.save(model.state_dict(), os.path.join(save_folder_name, path2weights))
    
    loss_func = train_utils.get_loss_func(train_params['loss'])
    opt = train_utils.get_opt(train_params["optimizer"], model, train_params["lr"] )
    lr_scheduler = train_utils.get_lr_sched(train_params['lr_scheduler'], opt)
    epochs_num = train_params["num_epochs"]

    now = datetime.now()
    now = now.strftime("%Y_%m_%d--%H_%M")

    
    # save json 
    json_object = json.dumps(config, indent=4)
    with open(os.path.join(log_folder_name, "config.json"), "w") as outfile:
        outfile.write(json_object)
    

    train_params = {
        "num_epochs": epochs_num,
        "optimizer": opt,
        "loss_func": loss_func,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "test_dl": test_dl,
        "sanity_check": False,
        "lr_scheduler": lr_scheduler,
        "path2weights": path2weights,
        "save_dir": save_folder_name,
        "logs_dir": log_folder_name,
        "name": name,
        "lr": train_params["lr"],
    }
    
    writer = SummaryWriter(log_folder_name)
    # TRAIN
    
    model, loss_hist, metric_hist = train_utils.train_val(model, train_params, writer)
    
    filename = os.path.join(log_folder_name, config["name"] )
    train_utils.plot_loss(loss_hist, metric_hist, filename)

def collate_fn_rnn(batch):
    imgs_batch, label_batch, names_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    labels_tensor = torch.stack(label_batch)
    
    return imgs_tensor, labels_tensor, names_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNN CNN')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args = parser.parse_args()

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    if args.config: # load config file
        with open(args.config) as handle:
            config = json.load(handle)
    elif args.resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        config = torch.load(args.resume)['config']
    else:
        sys.exit("Add config file")
    
    # with RNN, yes it should be temporal
    if config["task"] == 'temporal': 
        train_videos(config, args.resume)
    else:
        print(f'Please choose tasks.')
        
    
    




