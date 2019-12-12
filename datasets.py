import pickle
import json
import random

import torch as th
import numpy as np
from torch.utils.data import Dataset

import datasets_utils


'''
This module holds datasets for HPE task. 
'''

SEED = 0

PATH_H36M = '/cvlabdata2/cvlab/Human36m/OpenPose/'
PATH_MPII = '/cvlabsrc1/cvlab/datasets_victor/MPII_HumanPose/'

class H36M(Dataset):
    '''
    Provides H3.6M Dataset.
    '''
    def __init__(self, num_images=1000, mode='train', use_heatmaps=True, PATH_H36M=PATH_H36M):

        if not ((mode == 'train') or (mode == 'val')):
            raise Exception(
                'Incorrect mode type! Only "train" or "val" are available.')
        self.PATH_H36M = PATH_H36M
        self.mode = mode
        self.use_heatmaps = use_heatmaps
        self.orig_width = 256 if self.use_heatmaps else 224 # image shape after preprocessing
        self.map_width =  64

        # train: 1475888. #cams: 560. in cam min,max:  992,6339.
        # val:    417329. #cams: 175. in cam min,max: 1008,5873.


        # self.num_images = num_images
        self.desc = datasets_utils.compose_new_desc(num_images, mode)
        self.num_images = self.desc['num_total'] \
                        if self.desc['num_total'] < num_images else num_images

        pkl_path = self.PATH_H36M+mode+'_data.pkl'
        with open(pkl_path, 'rb') as f:
            self.pkl = pickle.load(f)

        self.get_descriptive_path = datasets_utils.get_desc_f(self.desc)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        scen, subj, cam, name = self.get_descriptive_path(idx)
        datapoint = self.pkl[scen][subj][cam][name]

        img_path = PATH_H36M+'S'+str(subj)+'/Images/'+cam+'_000000'+name+'.jpg'
        joints = th.tensor(datapoint['annotations_2d']).float()

        image, joints, joints_valid = datasets_utils.getitem(img_path, 
                                                             joints, 
                                                             self.use_heatmaps, 
                                                             self.map_width, 
                                                             self.orig_width)
        return image, joints, joints_valid

        # _3d_ = one_data['annotations_3d'].reshape(-1,3)
        # x_, y_, z_ = _3d_[:,0], _3d_[:,1], _3d_[:,2] 
        # fig,ax = plt.subplots(1,1,figsize=(10,10))
        # ax.set_xlim((-1000,1000))
        # ax.set_ylim((1000,-1000))

        # ax.plot(x_,y_, 'ro')
        # for i in range(len(_3d_)):
        # #     print(_3d_[i])
        #     plt.annotate(i, (_3d_[i,0], _3d_[i,1]))

class MPII(Dataset):
    '''
    Provides MPII Dataset.
    '''
    def __init__(self, num_images=1000, mode='train', use_heatmaps=True, PATH_MPII=PATH_MPII):

        self.use_heatmaps = use_heatmaps
        self.orig_width = 256 if self.use_heatmaps else 224
        self.map_width =  64

        if not ((mode == 'train') or (mode == 'val')):
            raise Exception(
                'Incorrect mode type! Only "train" or "val" are available.')

        self.PATH_MPII = PATH_MPII
        self.mode = mode
        self.num_images = num_images

        # mean = th.load(self.PATH_MPII+'Images/mean.pth')
        # self.mean = mean['mean']
        # self.std = mean['std']

        # As the number of images in MPII is relatively small (17408), 
        # we split its annotations into "train" and "val" sets.
        # Both lists of annotations are saved in the "datasets/MPII" folder.

        self.annotations = th.load('./datasets/MPII/'+mode+'_annotations.pth')
        self.annotations = self.annotations[:num_images]
        if len(self.annotations) < num_images:
            print('Warning: "num_images" exceeds available amount of data, {} instead.'.format(len(self.annotations)))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annot = self.annotations[idx]
        img_path = self.PATH_MPII + 'Images/images/' + annot['img_paths']
        joints = th.tensor([point[:2] for point in annot['joint_self']]).float() # 16 x 2
        joints = datasets_utils.map_joints(joints) # rename all joints acc. to the H36M protocol

        image, joints, joints_valid = datasets_utils.getitem(img_path, 
                                                             joints, 
                                                             self.use_heatmaps, 
                                                             self.map_width, 
                                                             self.orig_width)
        return image, joints, joints_valid







