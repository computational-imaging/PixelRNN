''' 9.9.22 Haley So
    - VideoDataset if using rnn
    - ImageDataset if using cnn
'''
import torch
import random
import glob
from PIL import Image
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from . import noise


# np.random.seed(2023)
# random.seed(2023)

class VideoDataset(Dataset):
    def __init__(self, ids, labels, transform, timesteps, labels2number, val=False, tulips=False, use_first=False):      
        self.transform = transform
        self.ids = ids
        self.labels = labels
        self.timesteps = timesteps
        self.labels_dict = labels2number
        self.path_label_dict = {}
        self.val_mode = val
        self.tulips=tulips
        self.use_first=use_first
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        path2imgs=glob.glob(self.ids[idx]+"/*.jpg")

        path2imgs.sort()
        if self.tulips:
            random.seed(2023)

        random.shuffle(path2imgs) 
        if self.timesteps > len(path2imgs):
            neededFrames = self.timesteps - len(path2imgs)
            if self.use_first:
                path2imgs.sort()
                first_frame= path2imgs[0]
                print(first_frame*neededFrames)
                sys.exit()
                path2imgs += neededFrames*first_frame
                print(path2imgs)
                sys.exit()
            else:
                # then we add frames to duplicate
                path2imgs = path2imgs + path2imgs[:neededFrames]

        if self.val_mode:
            # use set indicies            
            val_indices = np.linspace(0,len(path2imgs), self.timesteps, dtype=int, endpoint=False)
            available_offset = len(path2imgs)-1  - val_indices[-1]
            val_indices += available_offset
            path2imgs.sort()
            path2imgs_select = [path2imgs[i] for i in val_indices]
            # print(path2imgs_select)
        
        # else: # random sampling 
        #     path2imgs_select = path2imgs[:self.timesteps] 
        #     path2imgs_select.sort()
        else:

            train_indices = np.linspace(0,len(path2imgs), self.timesteps, dtype=int, endpoint=False)
            available_offset = len(path2imgs) - 1 - train_indices[-1]
            offset = random.randint(0,available_offset)
            train_indices += offset
            path2imgs.sort()
            path2imgs_select = [path2imgs[i] for i in train_indices]

        
        # random_choice = self.transform
        # label = self.labels_dict[self.labels[idx]]
        # frames = []
        # for p2i in path2imgs_select:
        #     frame = Image.open(p2i)
        #     frames.append(random_choice(frame))

        
        # label = self.labels_dict[self.labels[idx]]
        # to_tens = transforms.Compose([transforms.ToTensor()])
        # frames = []
        # for p2i in path2imgs_select:
        #     frame = Image.open(p2i)
        #     frames.append(frame)
        # frames = torch.stack(frames)
        # frames_tr = self.transform(frames)

        label = self.labels_dict[self.labels[idx]]
        frames = []
        for p2i in path2imgs_select:
            frame = Image.open(p2i)
            frames.append(frame)
        
        # seed = np.random.randint(1e9)        
        frames_tr = []
        for frame in frames:
            # random.seed(seed)
            # np.random.seed(seed)
            frame = self.transform(frame)
            frames_tr.append(frame)
            
        if len(frames_tr)>0:
            frames_tr = torch.stack(frames_tr)
        
        return frames_tr, label, name


class ImageDataset(Dataset):
    def __init__(self, ids, labels, transform, labels2number):      
        self.transform = transform
        self.ids = ids
        self.labels = labels
        self.labels_dict = labels2number
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        label_i = self.labels_dict[self.labels[idx]]
        image_i = Image.open(self.ids[idx])
        name_i = self.ids[idx]
        image_i = self.transform(image_i)
        return image_i, label_i, name_i
    

# input images are the ones from scamp
class ScampDataset(Dataset):
    def __init__(self, ids, labels, input_white, transform=None, pytorch_size=False, add_noise=False, mean_std=(0.0,0.0)):      
        self.ids = ids
        self.labels = labels
        self.input_white = input_white
        self.pytorch_size = pytorch_size
        self.transform = transform
        self.add_noise = add_noise
        self.mean = mean_std[0]
        self.std =  mean_std[1]
        
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        label_i = int(self.labels[idx])
        image_i = plt.imread(self.ids[idx])

        if self.pytorch_size:
            image_i = image_i[64:,:,1]
            # image_i = image_i[:64,:,1]
        else:
            image_i = image_i[128:192,128:192,1]

        image_i = (image_i-127.0) / self.input_white
        image_i = torch.Tensor(image_i)
        # image_i = torch.unsqueeze(image_i,0)
        image_i = torch.unsqueeze(image_i,0)
        if self.add_noise:
            image_i = image_i + torch.randn(image_i.size()) * self.std + self.mean
        if self.transform !=None:
            # print("adding transform")
            image_i = self.transform(image_i)

        name_i = self.ids[idx]
        return image_i, label_i, name_i
    

