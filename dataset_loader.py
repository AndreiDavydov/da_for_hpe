from matplotlib.pyplot import imread
import pickle
import json
import random

import torch as th
import numpy as np
from torch.utils.data import Dataset


'''
This module holds datasets for HPE task. 
'''

PATH_H36M = '/cvlabdata2/cvlab/Human36m/OpenPose/'
PATH_MPII = '/cvlabsrc1/cvlab/datasets_victor/MPII_HumanPose/'

class H36M(Dataset):
    '''
    Provides H3.6M Dataset.
    '''
    def __init__(self, PATH_H36M=PATH_H36M, num_images=1000, mode='train'):

        if not ((mode == 'train') or (mode == 'val')):
            raise Exception(
                'Incorrect mode type! Only "train" or "val" are available.')
        self.PATH_H36M = PATH_H36M
        self.mode = mode

        # train: 1475888. #cams: 560. in cam min,max:  992,6339.
        # val:    417329. #cams: 175. in cam min,max: 1008,5873.

        self.num_images = num_images

        pkl_path = self.PATH_H36M+mode+'_data.pkl'
        with open(pkl_path, 'rb') as f:
            self.pkl = pickle.load(f)

        scens_delete = []
        for scenario in self.pkl:
            if self.pkl[scenario] == {}:
                scens_delete.append(scenario)
        for scen in scens_delete:
            del self.pkl[scen]
        self.scenarios = list(self.pkl.keys())

        # there are no broken folders in train, 
        # but there is one in the val:
        # "Directions, 11, Directions.54138969".
        if self.mode == 'val':
            del self.pkl['Directions'][11]['Directions.54138969']

        self.subjects = (1,5,6,7,8) if mode == 'train' else (9,11)
        self.num_cameras = 8

    def __len__(self):
        return self.num_images

    def get_item(self, idx):
        ''' The problem is that it is hard to prepare dataset as a list 
        of datapaths. Hence, we propose the technique of getting i's sample
        based on the total number of images only.

        TODO!!!
        '''
        num_per_scen = int(self.num_images//
            len(self.scenarios))
        num_per_subj = int(self.num_images//
            (len(self.scenarios)*len(self.subjects)))
        num_per_cam = int(self.num_images//
            (len(self.scenarios)*len(self.subjects)*self.num_cameras)) 

        num_per_scen = 100 if num_per_scen == 0 else num_per_scen
        num_per_subj = 100 if num_per_subj == 0 else num_per_subj
        num_per_cam  = 100 if num_per_cam  == 0 else num_per_cam

        scen = idx // num_per_scen
        scen = len(self.scenarios) - 1 if scen >= len(self.scenarios) else scen
        idx -= scen * num_per_scen
        subj = idx // num_per_subj
        subj = len(self.subjects) - 1 if subj >= len(self.subjects) else subj
        idx -= subj * num_per_subj
        cam = idx // num_per_cam
        cam = self.num_cameras - 1 if cam >= self.num_cameras else cam
        idx -= cam * num_per_cam

        return scen, subj, cam, idx


    def __getitem__(self, idx):
        scen, subj, cam, idx = self.get_item(idx)

        scen = self.scenarios[scen]
        subj = self.subjects[subj]

        if scen == 'Directions' and subj == 11 and cam == 7:
            cam = 6
        cam = list(self.pkl[scen][subj].keys())[cam]
        name = list(self.pkl[scen][subj][cam].keys())[idx]
            
        datapoint = self.pkl[scen][subj][cam][name]

        img_path = PATH_H36M+'S'+str(subj)+'/Images/'+cam+'_000000'+name+'.jpg'
        image = th.tensor(imread(img_path)/255).permute(2,0,1)

        # heatmap = imread(PATH_H36M+datapoint['heatmap_path']).reshape(368,15,368)
        joints = th.tensor(datapoint['annotations_2d'])
        image, joints = prepare_input(image, joints)

        joints_valid = ~th.isnan(joints.sum(dim=1)) # to get same output as MPII
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
    def __init__(self, PATH_MPII=PATH_MPII, num_images=1000, mode='train', seed=0):

        self.seed = seed

        if not ((mode == 'train') or (mode == 'val')):
            raise Exception(
                'Incorrect mode type! Only "train" or "val" are available.')

        self.PATH_MPII = PATH_MPII
        self.mode = mode
        self.num_images = num_images

        mean = th.load(self.PATH_MPII+'Images/mean.pth')
        self.mean = mean['mean']
        self.std = mean['std']

        # As the number of images in MPII is relatively small (17408), 
        # we split its annotations into "train" and "val" sets.
        # Both lists of annotations are saved in the "datasets/MPII" folder.
        
        # annotations = json.load(open(PATH_MPII+'Annotations/mpii_annotations.json', 'rb'))
        # self.annotations, self.ids = self._initialize_annotations(annotations)

        self.annotations = th.load('./datasets/MPII/'+mode+'_annotations.pth')
        self.annotations = self.annotations[:num_images]

    def __len__(self):
        return len(self.annotations)

    # def _initialize_annotations(self, annotations):
    #     ''' Firstly, we follow "easy" strategy, processing MPII dataset.
    #     It likes to contain several people on same image, but keeping them
    #     in different "annotation" dictionaries. 
    #     We'd like to delete ALL non-unique annotations.
    #     Then shuffle these indices and cut the necessary part of them.
    #     '''

    #     # All this part till shuffle can be done with the preprocess!!!
    #     # TODO!!!
    #     unique_paths, ids_to_delete = [], []
    #     for idx,annot in enumerate(annotations):
    #         img_path = annot['img_paths']
    #         if img_path in unique_paths:
    #             ids_to_delete.append(idx)
    #         else:
    #             unique_paths.append(img_path)

    #     for idx in ids_to_delete[::-1]:
    #         del annotations[idx]

    #     # Now let's remain only "mode" data.
    #     mode_ids = []
    #     mode_key = 1 if self.mode == 'val' else 0
    #     for idx, annot in enumerate(annotations):
    #         if annot['isValidation'] == mode_key:
    #             mode_ids.append(idx)

    #     random.seed(self.seed)
    #     random.shuffle(mode_ids)
    #     mode_ids = mode_ids[:self.num_images]
    #     return annotations, mode_ids


    def __getitem__(self, idx):
        # old fashion getting item
        # annot = self.annotations[self.ids[idx]]
        annot = self.annotations[idx]
        img_path = self.PATH_MPII + 'Images/images/' + annot['img_paths']
        image = th.tensor((imread(img_path)/255)).permute(2,0,1)

        joint_self = annot['joint_self']
        joints = th.tensor([point[:2] for point in joint_self]) # 16 x 2
        joints = map_joints(joints) # rename all joints acc. to the H36M protocol

        joints_valid = joints.sum(dim=1) != 0 # 0,0 - non-visible points
        # joints = joints[joints_valid] # only for visualization

        image, joints = prepare_input(image, joints)

        return image, joints, joints_valid


def do_pad(img, joints):
    H,W = (int(img.shape[-2]), int(img.shape[-1]))
    
    if H > W:
        total_pad = H - W
        pad_l, pad_r = total_pad//2, total_pad - total_pad//2
        pad = (pad_l,pad_r, 0,0)
        img = th.nn.functional.pad(img, pad)
        joints[:,0] += pad_l
    elif W > H:
        total_pad = W - H
        pad_b, pad_t = total_pad//2, total_pad - total_pad//2
        pad = (0,0, pad_b,pad_t)
        img = th.nn.functional.pad(img, pad)  
        joints[:,1] += pad_b
    # otherwise, dimensions already coincide
    
    return img, joints

def prepare_input(img, joints):
    ''' Prepares data for ResNet50 architecture, 
    img_size = C x 224 x 224.


    Joints must change correspondingly.

    img - tensor, C x H x W
    joints - tensor, N x 2 # for the case of 2d HPE
    '''
    out_size = (224,224) # as ResNet50 requires


    joint_l, joint_r = int(joints[:,0].min()), int(joints[:,0].max())
    joint_b, joint_t = int(joints[:,1].min()), int(joints[:,1].max())

    H_box, W_box = int(joint_t-joint_b), int(joint_r-joint_l)
    center_box = (joint_l+W_box/2, joint_b+H_box/2)

    # img = C x H x W
    # computing box coordinates in the global system (img coordinates)
    box_l = int(np.maximum(center_box[0]-W_box, 0))
    box_r = int(np.minimum(center_box[0]+W_box, img.shape[-1]))
    box_t = int(np.minimum(center_box[1]+H_box, img.shape[-2]))
    box_b = int(np.maximum(center_box[1]-H_box, 0))

    # adjusting joints to the box
    joints[:,0] -= box_l
    joints[:,1] -= box_b

    img = img[..., box_b:box_t, box_l:box_r]

    # now let's pad image to make it square
    img, joints = do_pad(img, joints)
    
    # now scale img and joints to the "out_size"
    scale = img.shape[-1] / out_size[-1] # assuming scale is same for x and y
    img = th.nn.functional.interpolate(img.unsqueeze(0), out_size, 
            mode='bilinear',align_corners=True)[0]
    img.clamp_(0,1)
    joints /= scale

    return img, joints

def map_joints(joints_mpii):
    ''' Maps the MPII format of joints encoding to the 
    format of the H36M.
    0 -> 10; 1 -> 9; 2 -> 8; 3 -> 11; 4 -> 12; 5 -> 13;
    8 -> 1; 9 -> 0; 10 -> 4; 11 -> 3; 12 -> 2; 13 -> 5; 14 -> 6; 15 -> 7;
    (6 + 7) -> 14
    '''
    joints_h36m = joints_mpii.clone()[:-1]
    joints_h36m[[10,9,8,11,12,13]] = joints_mpii[[0,1,2,3,4,5]]
    joints_h36m[[1,0,4,3,2,5,6,7]] = joints_mpii[[8,9,10,11,12,13,14,15]]
    joints_h36m[14] = (joints_mpii[6] + joints_mpii[7])/2
    return joints_h36m

# def get_rect(l,r,t,b):
#     return [[l,l,r,r,l], 
#             [b,t,t,b,b]]











