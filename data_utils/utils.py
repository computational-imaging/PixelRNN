'''
    9/9/22 Haley So
    get_videos: 
    - get_videos
        I: (dataset)
        O: ids, labels, categories, and csv_file (for if there's a certain train/test csv)
    - split_data
        I: (videos, labels, num_classes, labels2number, csv_file=None, test_size=0.1)
        O: train_ids, train_labels, test_ids, test_labels 

'''
import os
from sklearn.model_selection import StratifiedShuffleSplit
import sys
import csv
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data_utils import dataset_classes


def images_dataset(dataset, resize_shape, batch_size, workers, config):
    if dataset =='mnist':
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(root='/media/data4b/haleyso/mnistdata', train=True, 
                transform=transforms.Compose([
                    # transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(resize_shape, ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomAffine(20, (1/14, 1/14)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,),(0.3081,)),
                    # transforms.Resize([resize_shape,resize_shape], interpolation=transforms.InterpolationMode.BILINEAR)
                ]), 
                download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
                datasets.MNIST(root='/media/data4b/haleyso/mnistdata', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,),(0.3081,)),
                    transforms.Resize([resize_shape,resize_shape], interpolation=transforms.InterpolationMode.BILINEAR),
                ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=True)
        num_classes = 10
    elif dataset =='cifar10':
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root='/media/data4b/haleyso/cifardata', train=True, 
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize([resize_shape,resize_shape], interpolation=transforms.InterpolationMode.BILINEAR)
                ]), 
                download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root='/media/data4b/haleyso/cifardata', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize([resize_shape,resize_shape], interpolation=transforms.InterpolationMode.BILINEAR),
                ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=True)
        num_classes = 10
    elif dataset == 'cambridge':
        
        # get dictionary of all images and their corresponding labels (folder names)
        # shuffle
        all_vids, all_labels, cats, csv_file = get_images(dataset)
        print("Number of videos: ", len(all_vids), " | Number of categories: ", len(cats))
        
        # number of classes we're using
        if config["train_params"]["num_classes"] != "all":
            num_classes = config["train_params"]["num_classes"]
        else:
            num_classes = len(cats)
        
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        # labels2number dictionary
        labels2number = {}
        cats.sort()
        for ind, uc in enumerate(cats):
            labels2number[uc] = ind 
        
        # split data
        # print("Number of classes we're using ", num_classes)

        train_ids, train_labels, val_ids, val_labels, test_ids, test_labels = split_data(all_vids, all_labels, num_classes, labels2number, csv_file=csv_file, val_size=0.1,test_size=0.1)
        print("Number of training: ", len(train_ids), " | Number of validation: ", len(val_ids), " | Number of test:", len(test_ids))
        # transformers:
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
        train_ds = dataset_classes.ImageDataset(ids= train_ids, labels= train_labels, transform= train_transformer,  labels2number=labels2number)
        val_ds = dataset_classes.ImageDataset(ids= val_ids, labels= val_labels, transform= test_transformer, labels2number=labels2number)
        test_ds = dataset_classes.ImageDataset(ids= test_ids, labels= test_labels, transform= test_transformer, labels2number=labels2number)

        # data_loaders
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size= batch_size, num_workers=workers, shuffle=True, pin_memory=True, drop_last=True) # does shuffle shuffle sequences??
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=True)  
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=True)  

    elif dataset == 'tulips1':
        
        # get dictionary of all images and their corresponding labels (folder names)
        # shuffle
        all_vids, all_labels, cats, csv_file = get_images(dataset)
        print("Number of videos: ", len(all_vids), " | Number of categories: ", len(cats))
        
        # number of classes we're using
        if config["train_params"]["num_classes"] != "all":
            num_classes = config["train_params"]["num_classes"]
        else:
            num_classes = len(cats)
        
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        # labels2number dictionary
        labels2number = {}
        cats.sort()
        for ind, uc in enumerate(cats):
            labels2number[uc] = ind 
        
        # split data
        # print("Number of classes we're using ", num_classes)
        # train_ids, train_labels,  test_ids, test_labels = split_data(all_vids, all_labels, num_classes, labels2number, csv_file)
        # print("Number of training: ", len(train_ids), " | Number of testing: ", len(test_ids))
        train_ids, train_labels, val_ids, val_labels, test_ids, test_labels = split_data(all_vids, all_labels, num_classes, labels2number, csv_file=csv_file, val_size=0.1,test_size=0.1)
        print("Number of training: ", len(train_ids), " | Number of validation: ", len(val_ids), " | Number of test:", len(test_ids))
        
        # transformers:
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
        train_ds = dataset_classes.ImageDataset(ids= train_ids, labels= train_labels, transform= train_transformer,  labels2number=labels2number)
        val_ds = dataset_classes.ImageDataset(ids= val_ids, labels= val_labels, transform= test_transformer, labels2number=labels2number)
        test_ds = dataset_classes.ImageDataset(ids= test_ids, labels= test_labels, transform= test_transformer, labels2number=labels2number)

        # data_loaders
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size= batch_size, num_workers=workers, shuffle=True, pin_memory=True, drop_last=True) # does shuffle shuffle sequences??
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=True)  
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=True)  
    else:
        sys.exit("Choose a dataset")
    
    return train_loader, val_loader, test_loader, num_classes

    

def get_images(dataset):
    if dataset == 'cambridge' or dataset == 'hmdb51' or dataset =='tulips1':
        if dataset == 'cambridge':
            path2data = '/media/data4b/haleyso/Cambridge_Hand_Gesture'
        elif dataset == 'tulips1':
            path2data = '/media/data4b/haleyso/tulips'
        else:
            path2data = "/media/data4b/haleyso/human_motion_data/hmdb51_jpg"

        listOfCats = os.listdir(path2data)
        ids = []
        labels = []

        for catg in listOfCats:
            path2catg = os.path.join(path2data, catg)
            listOfSubCats = os.listdir(path2catg)
            for video in listOfSubCats:
                video_folder = os.path.join(path2catg,video)
                list_images = os.listdir(video_folder)
                path2subCats= [os.path.join(video_folder,los) for los in list_images if not los.startswith('.')]
                ids.extend(path2subCats)
                labels.extend([catg]*len(path2subCats))
        csv_file=None
        # print(len(ids), len(labels), ids[1000], labels[1000])

    elif dataset == 'jester':
        csv_file = {}
        csv_file["train"]= "/media/data4b/haleyso/Jester/labels/train.csv"
        csv_file["test"]= "/media/data4b/haleyso/Jester/labels/test-answers.csv"
        csv_file["validation"]= "/media/data4b/haleyso/Jester/labels/validation.csv"
        csv_file['labels']= "/media/data4b/haleyso/Jester/labels/labels.csv"
        path2data = "/media/data4b/haleyso/Jester/20bn-jester-v1"

        ids = []
        labels = []
        listOfCats = []
        with open(csv_file['labels'], newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                row = (' '.join(row))
                listOfCats.append(row)
        
        for fil in ["train", "test", "validation"]:
            with open(csv_file[fil], newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    row = (' '.join(row))
                    id_, label_ = row.split(";")
                    ids.append(os.path.join(path2data, id_))
                    labels.append(label_)
    else:
        sys.exit("Please select a different dataset. ")

    return ids, labels, listOfCats, csv_file   


def get_videos(dataset):  
    # gets videos of either Cambridge, human motion data, or jester dataset
    # returns ids, labels, list of categories, csv_file

    if dataset == 'cambridge' or dataset == 'hmdb51' or dataset=='avletters' or dataset == 'tulips1' or dataset == 'egogesture':
        if dataset == 'cambridge':
            path2data = '/media/data4b/haleyso/Cambridge_Hand_Gesture'
        elif dataset == 'avletters':
            path2data = '/media/data4b/haleyso/avletters'
        elif dataset == 'tulips1':
            path2data = '/media/data4b/haleyso/tulips'
        elif dataset=='egogesture':
            path2data = '/media/data4b/haleyso/egogesture_processed'
        else:
            path2data = "/media/data4b/haleyso/human_motion_data/hmdb51_jpg"

        listOfCats = os.listdir(path2data)
        ids = []
        labels = []
        for catg in listOfCats:
            path2catg = os.path.join(path2data, catg)
            listOfSubCats = os.listdir(path2catg)
            path2subCats= [os.path.join(path2catg,los) for los in listOfSubCats]
            ids.extend(path2subCats)
            labels.extend([catg]*len(listOfSubCats))
        if dataset == 'egogesture':
            train_ids = [3, 4, 5, 6, 8, 10, 15, 16, 17, 20, 21, 22, 23, 25, 26, 27, 30, 32, 36, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50]
            val_ids = [1, 7, 12, 13, 24, 29, 33, 34, 35, 37]
            test_ids= [2, 9, 11, 14, 18, 19, 28, 31, 41, 47]
            train_subjs = ['subject'+str(train_id).zfill(2) for train_id in train_ids]
            val_subjs = ['subject'+str(val_id).zfill(2) for val_id in val_ids]
            test_subjs = ['subject'+str(test_id).zfill(2) for test_id in test_ids]

            csv_file = [train_subjs, val_subjs, test_subjs]
        else:
            csv_file=None

    elif dataset == 'jester':
        csv_file = {}
        csv_file["train"]= "/media/data4b/haleyso/Jester/labels/train.csv"
        csv_file["test"]= "/media/data4b/haleyso/Jester/labels/test-answers.csv"
        csv_file["validation"]= "/media/data4b/haleyso/Jester/labels/validation.csv"
        csv_file['labels']= "/media/data4b/haleyso/Jester/labels/labels.csv"
        csv_file['path2data'] = "/media/data4b/haleyso/Jester/20bn-jester-v1"

        ids = []
        labels = []
        listOfCats = []
        with open(csv_file['labels'], newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                row = (' '.join(row))
                listOfCats.append(row)
        
        for fil in ["train", "test", "validation"]:
            with open(csv_file[fil], newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    row = (' '.join(row))
                    id_, label_ = row.split(";")
                    ids.append(os.path.join(csv_file['path2data'], id_))
                    labels.append(label_)
    

    else:
        sys.exit("Please select a different dataset. ")

    return ids, labels, listOfCats, csv_file



def split_data(videos, labels, num_classes, labels2number, csv_file=None, val_size=0.1, test_size=0.1):
    # splits into train_ids, train_labels, test_ids, test_labels. 10% test size if csv file not given

    if num_classes == None:
        sys.exit("Oops, using no classes")
    ids = [id_ for id_, label in zip(videos, labels) if labels2number[label]<num_classes]
    labels = [label for id_, label in zip(videos, labels) if labels2number[label]<num_classes]
    
    # Dataset Splits
    if csv_file==None:
        strat = StratifiedShuffleSplit(n_splits=2, test_size=test_size+val_size, random_state=0)
        
        # strat = StratifiedShuffleSplit(n_splits=2, test_size=.15, random_state=0)
        train_indx, testing_indx = next(strat.split(ids, labels))
        train_ids = [ids[i] for i in train_indx]
        train_labels = [labels[i] for i in train_indx]

        testing_ids = [ids[i] for i in testing_indx]
        testing_labels = [labels[i] for i in testing_indx]

        
        
        strat2 = StratifiedShuffleSplit(n_splits=2, test_size=.5, random_state=0)
        val_indx, test_indx = next(strat2.split(testing_ids, testing_labels))

        val_ids = [testing_ids[i] for i in val_indx]
        val_labels = [testing_labels[i] for i in val_indx] 
        test_ids = [testing_ids[i] for i in test_indx]
        test_labels = [testing_labels[i] for i in test_indx] 

        return train_ids, train_labels, val_ids, val_labels, test_ids, test_labels 

        # strat = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=0)
        # train_indx, test_indx = next(strat.split(ids, labels))

        # train_ids = [ids[i] for i in train_indx]
        # train_labels = [labels[i] for i in train_indx]
        # test_ids = [ids[i] for i in test_indx]
        # test_labels = [labels[i] for i in test_indx]
        # return train_ids, train_labels, test_ids, test_labels 
    elif type(csv_file) is dict:

        train_ids = []
        train_labels = []
        val_ids = []
        val_labels = []
        test_ids = []
        test_labels = []

        
        with open(csv_file["train"], newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                row = (' '.join(row))
                id_, label_ = row.split(";")
                train_ids.append(os.path.join(csv_file['path2data'], id_))
                train_labels.append(label_)
        with open(csv_file["validation"], newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                row = (' '.join(row))
                id_, label_ = row.split(";")
                val_ids.append(os.path.join(csv_file['path2data'], id_))
                val_labels.append(label_)

        with open(csv_file["test"], newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                row = (' '.join(row))
                id_, label_ = row.split(";")
                test_ids.append(os.path.join(csv_file['path2data'], id_))
                test_labels.append(label_)
        return train_ids, train_labels, val_ids, val_labels, test_ids, test_labels 
    else:
        print("egogesture dataset")
        train_subjs = csv_file[0]
        val_subjs = csv_file[1]
        test_subjs = csv_file[2]

        #     ids = [id_ for id_, label in zip(videos, labels) if labels2number[label]<num_classes]
        #    labels = [label for id_, label in zip(videos, labels) if labels2number[label]<num_classes]

        train_ids = []
        train_labels = []
        val_ids = []
        val_labels = []
        test_ids = []
        test_labels = []
        
        for i, id_i in enumerate(ids):
            # print(id_i, labels[i])
            name = id_i.split('/')[-1]
            subj_i = name.split('S')[0]
            if subj_i in train_subjs:
                train_ids.append(id_i)
                train_labels.append(labels[i])
                
            elif subj_i in val_subjs:
                val_ids.append(id_i)
                val_labels.append(labels[i])

            elif subj_i in test_subjs:
                test_ids.append(id_i)
                test_labels.append(labels[i])

            else:
                print("utils.py: error with egogesture split")
                sys.exit()

    return train_ids, train_labels, val_ids, val_labels, test_ids, test_labels 