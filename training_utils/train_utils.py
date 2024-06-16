import torch
import copy
from torchvision.transforms.functional import to_pil_image
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import matplotlib.pylab as plt
from tqdm import tqdm 
from torch import nn
import sys
import os




def denormalize(x_, mean, std):
    x = x_.clone()
    for i in range(3):
        x[i] = x[i]*std[i]+mean[i]
    x = to_pil_image(x)        
    return x

def train_val(model, train_params, writer, linear_only=False):
# def train_val(model, params, method=None): 
    num_epochs=train_params["num_epochs"]
    loss_func=train_params["loss_func"]
    opt=train_params["optimizer"]
    train_dl=train_params["train_dl"]
    val_dl=train_params["val_dl"]
    test_dl=train_params["test_dl"]
    sanity_check=train_params["sanity_check"]
    lr_scheduler=train_params["lr_scheduler"]
    path2weights=train_params["path2weights"]
    save_dir=train_params["save_dir"]
    logs_dir=train_params["logs_dir"]
    name = train_params['name']
    lr = train_params['lr']

    ensure_dir(save_dir)
    # print(f'name to save to {os.path.join(save_dir, name+".pt")}')
    # sys.exit()

    loss_history={
        "train": [],
        "val": [],
        "test": [],
    }
    
    metric_history={
        "train": [],
        "val": [],
        "test": [],
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    print("LOADING WEIGHTS")
    model.load_state_dict(torch.load(os.path.join(save_dir, path2weights)))
    with torch.no_grad():
        model.eval()
        test_loss, test_metric = loss_epoch(model,loss_func,test_dl,sanity_check,linear_only=linear_only)  
        best_loss, best_metric = loss_epoch(model,loss_func,val_dl,sanity_check,linear_only=linear_only)  
    corresponding_test = test_metric

    print("The best loss so far is ", best_loss, " | The best accuracy ", 100*best_metric)
    for epoch in range(num_epochs):
        current_lr=get_lr(opt)
        
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt, linear_only=linear_only)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_metric, epoch)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        model.eval()
        if epoch>1:
            eval_val=False
        with torch.no_grad():

            if (epoch == num_epochs-1):
                print_outputs=True
            else:
                print_outputs=False
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl,sanity_check,eval_val=True,san_name=0, linear_only=linear_only)
            test_loss, test_metric=loss_epoch(model,loss_func,test_dl,sanity_check,eval_val=True,san_name=0, linear_only=linear_only, print_outputs=print_outputs)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Acc/val", val_metric, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Acc/test", test_metric, epoch)
        if val_loss <= best_loss:
            best_loss = val_loss
        if val_metric > best_metric:
            # Save the best model based on validation 
            best_metric = val_metric
            corresponding_test = test_metric # this is the test metric corresponding to the best validation model
            best_model_wts = copy.deepcopy(model.state_dict())
            save_to_file = os.path.join(save_dir, name+".pt")
            torch.save(model.state_dict(),  save_to_file)
            print(f"train acc: {train_metric} | val acc : {val_metric} | test acc: {test_metric}")


            
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        loss_history["test"].append(test_loss)
        metric_history["test"].append(test_metric)
        
        lr_scheduler.step(val_loss)


        print("train loss: %.6f, val loss: %.6f,  train_accuracy: %.2f, val accuracy: %.2f, best val acc so far: %.2f" %(train_loss,val_loss,100*train_metric,100*val_metric, 100*best_metric))
        print("test ac: %.2f | test metric corresponding to best val model: %.2f  " %(100*test_metric, 100*corresponding_test))
        print("-"*10) 


    torch.save(model.state_dict(), f"{save_dir}/last_{name}_lr_{lr}_test_{test_metric}.pt")
    print(f"saved to: {save_dir}/last_{name}_lr_{lr}_test_{test_metric}.pt")
    model.load_state_dict(best_model_wts)
    
    writer.flush()
    writer.close()
    return model, loss_history, metric_history

# get learning rate 
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def get_lr_sched(sched, opt):
    if sched == "ReduceLROnPlateau":
        lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=5,verbose=1)
    else:
        sys.exit("Need to define scheduler")
    return lr_scheduler

def get_opt(optimi, model, lr):
    if optimi =="Adam":
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        sys.exit("Need to define optimizer")
    return opt

def get_loss_func(loss_type):
    if loss_type == "cross_entropy":
        loss = nn.CrossEntropyLoss(reduction="sum")
        loss = loss.cuda()
    else:
        sys.exit("Need to define loss function")
    return loss

def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None, use_weight=False, model=None, original_weights=None):
    loss = loss_func(output, target)
    # print(loss)
    # sys.exit()
    if use_weight:
        # print('using weight')
        mse = nn.MSELoss()
        weight_loss = mse(model.fc.weight, original_weights)
        hp = 1
        # print(f'hp: {hp} loss {loss} weight_loss {weight_loss}')
        loss = hp * weight_loss + loss
    with torch.no_grad():
        metric_b = metrics_batch(output,target)
    if opt is not None:

        opt.zero_grad()
        loss.backward()
        opt.step()
        # print("oki")
        # sys.exit()
    return loss.item(), metric_b
    

def loss_epoch(model,loss_func,dataset_dl, sanity_check=False,opt=None, print_intermediates=False, save_dir=None, eval_val=False, san_name=0, min_max=(0.0,0.0), linear_only=False, use_weight=False, original_weights=None, input_white=5, print_outputs=False):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)
    num_i = 0
    print(len_data, "length of data")
    for xb, yb, name in dataset_dl:
    # for xb, yb, name in tqdm(dataset_dl):
        xb=xb.cuda()
        yb=yb.cuda()

        if (eval_val):
            None
        #     if num_i ==0:
        #         print(xb.size())

        #         immy0 = xb[0,0,0,:,:].cpu().detach().squeeze()
        #         immy1 = xb[1,0,0,:,:].cpu().detach().squeeze()
        #         immy2 = xb[2,0,0,:,:].cpu().detach().squeeze()
        #         immy3 = xb[3,0,0,:,:].cpu().detach().squeeze()
        #         immy = torch.hstack([immy0,immy1, immy2, immy3])
        #         print(immy.size())
        #         savname = str(san_name)+".BMP"
        #         print(os.path.join("/home/haleyso/CNN_CGRU/",savname))
        #         plt.imsave(os.path.join("/home/haleyso/CNN_CGRU/",savname), immy.numpy(), vmin=-5, vmax=5, cmap='gray')
        #         # numero +=1

        #         # sys.exit()
        #         num_i +=1
            

        if print_intermediates: #batch_size = 1
            vidname = name[0].split('/')[-2:]
            save_path = os.path.join(save_dir, vidname[0],vidname[1])

        else:
            save_path = save_dir
        
        output=model(xb, print_intermediates=print_intermediates, save_dir=save_path, min_max=min_max, linear_only=linear_only, input_white=input_white)
        
        
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt, use_weight, model, original_weights)
        running_loss+=loss_b
        
        if metric_b is not None:
            running_metric+=metric_b
        if sanity_check is True:
            break
        if print_outputs:
            in_mean = xb.mean().cpu().detach().numpy()
            print(f'name: {name}, xb.mean {in_mean} ')
            print(output.cpu().detach().numpy(), torch.argmax(output).cpu().detach().numpy(), yb.cpu().detach().numpy())
    loss=running_loss/float(len_data)
    metric=running_metric/float(len_data)
    # print(f"model training status: {model.training}")


    return loss, metric


def plot_loss(loss_hist, metric_hist, filename):

    num_epochs= len(loss_hist["train"])

    # plt.title("Train-Val Loss")
    # plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
    # plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
    # plt.ylabel("Loss")
    # plt.xlabel("Training Epochs")
    # plt.legend()
    # plt.show()
    # plt.savefig(filename+"loss.png")

    plt.title("Train-Val Accuracy")
    plt.plot(range(1,num_epochs+1), metric_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1), metric_hist["val"],label="val")
    plt.plot(range(1,num_epochs+1), metric_hist["test"],label="test")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()
    plt.savefig(filename+"_accuracy.png")



def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def scamp_shape(image):
        row1 = torch.cat((image[:,0:1,:,:],image[:,1:2,:,:],image[:,2:3,:,:],image[:,3:4,:,:]), 2)
        row2 = torch.cat((image[:,4:5,:,:],image[:,5:6,:,:],image[:,6:7,:,:],image[:,7:8,:,:]), 2)
        row3 = torch.cat((image[:,8:9,:,:],image[:,9:10,:,:],image[:,10:11,:,:],image[:,11:12,:,:]), 2)
        row4 = torch.cat((image[:,12:13,:,:],image[:,13:14,:,:],image[:,14:15,:,:],image[:,15:16,:,:]), 2)
        scamp_image = torch.cat((row1,row2,row3,row4),3) # 4,1,64,64

        return scamp_image


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)