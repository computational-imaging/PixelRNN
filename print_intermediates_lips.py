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

seed = 55
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.set_num_threads(1)
# torch.use_deterministic_algorithms(True)

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

def test_scamp_linear(config, save_dir, replace_h_weight=False, save_images=False, input_white=5):
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
    # train_ids, train_labels, test_ids, test_labels = utils.split_data(all_vids, all_labels, num_classes, labels2number, csv_file=csv_file, test_size=0.1)
    # print("Number of training: ", len(train_ids), " | Number of testing: ", len(test_ids))
    train_ids, train_labels, val_ids, val_labels, test_ids, test_labels = utils.split_data(all_vids, all_labels, num_classes, labels2number, csv_file=csv_file, val_size=0.1,test_size=0.1)
    print("Number of training: ", len(train_ids), " | Number of validation: ", len(val_ids), " | Number of test:", len(test_ids))
    # test videos are in test_ids

    # toprint=[]
    # for name in test_ids:
    #     topr = name.split("/")[-1]
    #     toprint.append(topr)
    # toprint.sort()
    # print(toprint)
    # sys.exit()

    # resize shape
    if cnn_model_type in resize_dict.keys():
        resize_shape = resize_dict[cnn_model_type]
    else:
        resize_shape = 64
    print(resize_shape)
    
    train_transformer = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),  
        # transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),    
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
    # test_ds = dataset_classes.VideoDataset(ids= test_ids, labels= test_labels, transform= test_transformer, timesteps=timesteps, labels2number=labels2number, val=True)
    # test_dl = DataLoader(test_ds, batch_size= batch_size, num_workers=4, shuffle=False, collate_fn= collate_fn_rnn)  
    if ("duplicate_first_frame" in train_params.keys()):
        use_first = train_params["duplicate_first_frame"]
    else:
        use_first = False

    train_ds = dataset_classes.VideoDataset(ids= train_ids, labels= train_labels, transform= train_transformer, timesteps=timesteps, labels2number=labels2number,  val=False, use_first=use_first)
    train_ds_test = dataset_classes.VideoDataset(ids= train_ids, labels= train_labels, transform= test_transformer, timesteps=timesteps, labels2number=labels2number,  val=False, use_first=use_first)

    val_ds = dataset_classes.VideoDataset(ids= val_ids, labels= val_labels, transform= test_transformer, timesteps=timesteps, labels2number=labels2number,val=True, tulips=True, use_first=use_first)
    test_ds = dataset_classes.VideoDataset(ids= test_ids, labels= test_labels, transform= test_transformer, timesteps=timesteps, labels2number=labels2number,val=True, tulips=True, use_first=use_first)



    # data_loaders
    train_dl = DataLoader(train_ds, batch_size= 1, num_workers=4, shuffle=True, collate_fn= collate_fn_rnn) # does shuffle shuffle sequences??
    train_test_dl = DataLoader(train_ds_test, batch_size= 1, num_workers=4, shuffle=False, collate_fn= collate_fn_rnn) # does shuffle shuffle sequences??
    
    val_dl = DataLoader(val_ds, batch_size= 1, num_workers=4, shuffle=False, collate_fn= collate_fn_rnn)  
    test_dl = DataLoader(test_ds, batch_size= 1, num_workers=4, shuffle=False, collate_fn= collate_fn_rnn)  

    mini = 0
    maxi = 0

    for im, lab, nam in test_dl:
        if im.min() < mini:
            mini = im.min()
        if im.max() > maxi:
            maxi = im.max()
    print(f'mini {mini} and maxi {maxi}')
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
    # if ("dropout" in train_params.keys()):
    #     dropout = train_params["dropout"]
    # else:
    #     dropout = 0.0

    # model = get_model.get_model(config, cnn_model_type, cnn_parameters, rnn_model_type, rnn_parameters, dataset, num_classes, max_pool, dropout=dropout)
    model = get_model.get_model(config, cnn_model_type, cnn_parameters, rnn_model_type, rnn_parameters, dataset, num_classes, max_pool)

    # print(model)
    # sys.exit()
    model = model.cuda()
    path2weights = config["path2weights"]
    print(path2weights)
    checkpoint = torch.load(path2weights)
    model.load_state_dict(checkpoint)
    model.eval()

    # replace the weight
    if replace_h_weight:
        model_params = model.state_dict()
        with torch.no_grad():
            model_params["rnn_model.Conv_out_h.weight"] = model_params["rnn_model.Conv_out_i.weight"]
            model.load_state_dict(model_params)

    loss_func = train_utils.get_loss_func(train_params['loss'])
    test_loss, test_metric = train_utils.loss_epoch(model,loss_func,test_dl, print_intermediates=save_images, save_dir=save_dir, min_max=(mini, maxi), input_white=input_white, print_outputs=True)
    train_loss, train_metric = train_utils.loss_epoch(model,loss_func,train_dl, print_intermediates=False, save_dir=save_dir, min_max=(mini, maxi), input_white=input_white)
    train_test_loss, train_test_metric = train_utils.loss_epoch(model,loss_func,train_test_dl)
    val_loss, val_metric = train_utils.loss_epoch(model,loss_func,val_dl)
    print(f'Test Loss: {test_loss} and Acc: {test_metric}')
    print(f'Train Loss: {train_loss} and Acc: {train_metric}')
    print(f'Train Loss with no transforms: {train_test_loss} and Acc: {train_test_metric}')
    print(f'Val Loss: {val_loss} and Acc: {val_metric}')
    print(f'Intermediates saved to {save_dir}')



def collate_fn_rnn(batch):
    imgs_batch, label_batch, names_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor, names_batch



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PixelRNN save out intermediate outputs')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-sd', '--save_dir', default=None, type=str, help='scamp file path (default: None)')
    parser.add_argument('-si', '--save_images', default=False, type=bool, help='Do you want to save our the intermediate outputs?')
    parser.add_argument('-rh', '--replace_h_weight', default=False, type=bool, help='Do we want to fix the gate in the rnn and make the h weight equal to the i weight?')
    parser.add_argument('-iw', '--input_white', default=10, type=float, help='input white value. default is 10.')
    args = parser.parse_args()

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    if args.config: # load config file
        with open(args.config) as handle:
            config = json.load(handle)
        train_utils.ensure_dir(args.save_dir)
    else:
        sys.exit("Add config file")
    
    # input_white = 10
    test_scamp_linear(config, args.save_dir, args.replace_h_weight, args.save_images, args.input_white)

    
    




