from pyexpat import model
from model_utils import cnn, rnn, model_archs
import torch
import sys
from training_utils import train_utils, quantization

cnn_names = sorted(name for name in cnn.__dict__ 
                     if callable(cnn.__dict__[name]))
rnn_names = sorted(name for name in rnn.__dict__ 
                     if callable(rnn.__dict__[name]))


def str_to_class(classname):
    print(getattr(sys.modules[__name__], classname))
    return getattr(sys.modules[__name__], classname)

def get_model(config, cnn_model_type, cnn_parameters, rnn_model_type, rnn_parameters, dataset, num_classes, max_pool, bfloat16=False, dropout=None):
    '''    
    cnn_parameters= {
        "dr_rate": cnn_params['dr_rate'],
        "kernel_size": cnn_params['kernel_size'],
        "stride": cnn_params['stride'],
        "dilation": cnn_params['dilation'],
        "groups": cnn_params['groups'],
        "bias": cnn_params['bias'],
        "cnn_method": train_params["cnn_method"],                               # weights quantization method
        "cnn_output_quantization" : train_params["cnn_output_quantization"],    # result quantization method
        "final_conv_method": train_params["final_conv_method"]                  # final convolution weight quantization method
    }   

    # rnn model
    rnn_parameters = {
        "kernel_size": rnn_params['kernel_size'],
        "stride": rnn_params['stride'],
        "dilation": rnn_params['dilation'],
        "groups": rnn_params['groups'],
        "bias": rnn_params['bias'],
        "rnn_num_layers": rnn_params['rnn_num_layers'],
        "rnn_hidden_size": rnn_params['rnn_hidden_size'],
        "method": train_params["rnn_method"],
        "hidden_quantization" : train_params["hidden_quantization"],            # hidden state quantization method
        "hidden_weight_init_scale": rnn_params['hidden_weight_init_scale']

    }
'''

    if "model_type" in config.keys():
        model_type = config["model_type"]
    else:
        model_type = "1_lin"

    if cnn_model_type == 'CNN_1_8':
        eight_channels=True
    else:
        eight_channels=False


    cnn_flag=True
    rnn_flag=True
    if cnn_model_type == 'FULL_linear':
        model = model_archs.FULL_linear(num_classes)
        return model
    if cnn_model_type == 'raw_cam':
        if model_type == '2_lin':
            model = model_archs.raw_cam_2lin(num_classes) #--------------------------------------------------------------------diff_cam_2lin raw_cam_2lin
        else:
            model = model_archs.raw_cam(num_classes)
        return model
    if cnn_model_type == 'raw_cam_downsampled':
        model = model_archs.raw_cam_downsampled(num_classes)
        return model
    if cnn_model_type == 'raw_cam_downsampled_bin_in':
        cnn_method = quantization.get_method_func(cnn_parameters['cnn_method'])
        model = model_archs.raw_cam_downsampled_bin_in(num_classes, cnn_method)
        return model
    if cnn_model_type == 'event_cam':
        model = model_archs.event_cam(num_classes)
        return model
    if cnn_model_type == 'diff_cam':
        if model_type == '2_lin':
            model = model_archs.diff_cam_2lin(num_classes)
        else:
            model = model_archs.diff_cam(num_classes)
        return model
    if cnn_model_type == 'nstd':
        if model_type == '2_lin':
            model = model_archs.nstd_2lin(num_classes)
        else:
            model = model_archs.nstd_1lin(num_classes)
        return model
    if cnn_model_type == 'piotr_event_cam': # evaluates last frames
        model = model_archs.piotr_event_cam(num_classes)
        return model
    if cnn_model_type == 'piotr_event_cam_spatial': # evaluates every frame
        model = model_archs.piotr_event_cam_spatial(num_classes)
        return model
    if cnn_model_type == 'CAM_THAT_CNNs_ORIGINAL':
        print("LAURIE ORIGINAL")
        cnn_method = quantization.get_method_func('laurie_ternary')
        linear_method = quantization.get_method_func('laurie_ternary')
        model = model = model_archs.CAM_THAT_CNNs_ORIGINAL(cnn_method, linear_method, num_classes)
        return model

    if cnn_model_type not in cnn_names:
        print("No cnn model was selected:")
        cnn_flag=False
    else:
        cnn_method = quantization.get_method_func(cnn_parameters['cnn_method'])
        
        print(cnn_parameters.get('cnn_output_quantization'))
        cnn_output_quantization = quantization.get_method_func(cnn_parameters.get('cnn_output_quantization'))

        cnn_model = cnn.__dict__[cnn_model_type](cnn_parameters, cnn_method, cnn_output_quantization, max_pool )
        if bfloat16:
            cnn_model = cnn_model.to(dtype=torch.bfloat16)
            # cnn_model = cnn_model.to(dtype=torch.half)
        # cnn_model = torch.nn.DataParallel(cnn.__dict__[cnn_model_type](cnn_parameters, cnn_method ))


    if rnn_model_type not in rnn_names:
        print("No rnn model was selected:")
        rnn_flag=False
    else:
        rnn_method = quantization.get_method_func(rnn_parameters['method'])
        hidden_quantization = quantization.get_method_func(rnn_parameters['hidden_quantization'])
        gate_quantization = quantization.get_method_func(rnn_parameters['gate_quantization'])
        rnn_model = rnn.__dict__[rnn_model_type](rnn_parameters, rnn_method, hidden_quantization, gate_quantization, max_pool )
        if bfloat16:
            rnn_model = rnn_model.to(dtype=torch.bfloat16)
            # rnn_model = rnn_model.to(dtype=torch.half)
        # rnn_model = torch.nn.DataParallel(rnn.__dict__[rnn_model_type](rnn_parameters, rnn_method )) 


    final_conv_method = quantization.get_method_func(cnn_parameters['final_conv_method'])
    if model_type == '1_cnn_1_lin':
        if cnn_flag and rnn_flag:
            # print("doing the new ")
            # sys.exit()
            model = model_archs.CNN_RNN_CNN_linear(cnn_model, cnn_parameters, rnn_model, rnn_parameters, final_conv_method, num_classes, max_pool, eight_channels)
        else:
            print("Need to implement the 1cnn1lin for cnn and rnn onlys")
            sys.exit()
        # elif cnn_flag and not rnn_flag:
        #     model = model_archs.CNN_linear(cnn_model, cnn_parameters, final_conv_method, num_classes, max_pool)
        # elif not cnn_flag and rnn_flag:
        #     rnn_model = rnn.__dict__[rnn_model_type](rnn_parameters, rnn_method, hidden_quantization, gate_quantization, max_pool, rnn_only_flag=True )
        #     model = model_archs.RNN_linear(rnn_model, rnn_parameters, final_conv_method, num_classes, max_pool)
    elif model_type == '2_lin':
        if cnn_flag and rnn_flag:
            model = model_archs.CNN_RNN_linear_linear(cnn_model, cnn_parameters, rnn_model, rnn_parameters, final_conv_method, num_classes, max_pool, eight_channels)
        else:
            print("Need to implement the 2lin for cnn and rnn onlys")
            sys.exit()
    else: 
        if cnn_flag and rnn_flag:
            if bfloat16:
                model = model_archs.CNN_RNN_linear(cnn_model, cnn_parameters, rnn_model, rnn_parameters, final_conv_method, num_classes, max_pool, eight_channels, bfloat=True, dropout=dropout)
            else:
                model = model_archs.CNN_RNN_linear(cnn_model, cnn_parameters, rnn_model, rnn_parameters, final_conv_method, num_classes, max_pool, eight_channels, dropout=dropout)
        elif cnn_flag and not rnn_flag:
            if config['task'] == 'temporal':
                model = model_archs.cnn1_1lin(cnn_model, cnn_parameters, final_conv_method, num_classes, max_pool, eight_channels)
            else:
                model = model_archs.CNN_linear(cnn_model, cnn_parameters, final_conv_method, num_classes, max_pool, eight_channels)
        elif not cnn_flag and rnn_flag:
            rnn_model = rnn.__dict__[rnn_model_type](rnn_parameters, rnn_method, hidden_quantization, gate_quantization, max_pool, rnn_only_flag=True )
            model = model_archs.RNN_linear(rnn_model, rnn_parameters, final_conv_method, num_classes, max_pool, eight_channels)
        else:
            print("oh no. see get_model.py line 99")
            sys.exit()
    if bfloat16:
        print("converting to bfloat16")
        model = model.to(dtype=torch.bfloat16)
        # model = model.to(dtype=torch.half)
    return model