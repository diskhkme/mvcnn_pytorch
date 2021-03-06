import numpy as np
import glob
import torch.utils.data
import os
import math
from skimage import io, transform
from PIL import Image
import torch
import torchvision as vision
from torchvision import transforms, datasets
from tools.augmentation import RandomRemoval
import random

class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12, shuffle=True, KNU_data=True,
                 pixel_augmentation=False):
        if KNU_data:
            self.classnames = ['BlindFlange', 'Cross', 'Elbow 90', 'Elbow non 90', 'Flange', 'Flange WN',
                               'Olet', 'OrificeFlange', 'Pipe', 'Reducer CONC', 'Reducer ECC',
                               'Reducer Insert', 'Safety Valve', 'Strainer', 'Tee', 'Tee RED',
                               'Valve', 'Wye']
        else:
            self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                             'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                             'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                             'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                             'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            if KNU_data:
                all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*_v*.png'))
                all_files = [file.replace('\\', '/') for file in all_files]
            else:
                all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*shaded*.png'))
                all_files = [file.replace('\\', '/') for file in all_files]

            ## Select subset for different number of views
            stride = int(12/self.num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new


        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        else:
            if pixel_augmentation == False:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    RandomRemoval(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])


    def __len__(self):
        return int(len(self.filepaths)/self.num_views)


    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])



class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12, KNU_data=True,pixel_augmentation=False):
        if KNU_data:
            self.classnames = ['BlindFlange', 'Cross', 'Elbow 90', 'Elbow non 90', 'Flange', 'Flange WN',
                               'Olet', 'OrificeFlange', 'Pipe', 'Reducer CONC', 'Reducer ECC',
                               'Reducer Insert', 'Safety Valve', 'Strainer', 'Tee', 'Tee RED',
                               'Valve', 'Wye']
        else:
            self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                             'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                             'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                             'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                             'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            if KNU_data:
                all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*_v*.png'))
                all_files = [file.replace('\\', '/') for file in all_files]
            else:
                all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*shaded*.png'))
                all_files = [file.replace('\\','/') for file in all_files]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if pixel_augmentation == False:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                RandomRemoval(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)

        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)

        return (class_id, im, path)

