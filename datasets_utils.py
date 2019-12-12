import torch as th
import numpy as np

# from matplotlib.pyplot import imread
from PIL import Image
from torchvision import transforms

SIGMA = 1

def compose_new_desc(num_images, mode):
    desc = th.load('./datasets/H36M/'+mode+'_desc.pth')
    num_total = desc['num_total']
    step = int(np.ceil(num_total / num_images))
    new_desc = {'num_total':num_images, 'num_per_scen':{}, 'num_per_subj':{}, 'num_per_cam':{}}

    # get "num_per_cam"
    new_num_total = 0
    for scen_name in desc['num_per_cam']:
        scen = desc['num_per_cam'][scen_name]
        new_desc['num_per_cam'][scen_name] = {}
        for subj_name in scen:
            subj = scen[subj_name]
            new_desc['num_per_cam'][scen_name][subj_name] = {}
            for cam_name in subj:
                cam = subj[cam_name]
                new_cam = int(np.ceil(cam/step))
                new_num_total += new_cam
                new_desc['num_per_cam'][scen_name][subj_name][cam_name] = new_cam

    # get "num_total"
    new_desc['num_total'] = new_num_total

    # get "num_per_subj"
    for scen_name in new_desc['num_per_cam']:
        scen = new_desc['num_per_cam'][scen_name]
        new_desc['num_per_subj'][scen_name] = {}
        for subj_name in scen:
            subj = scen[subj_name]
            new_desc['num_per_subj'][scen_name][subj_name] = 0
            for cam_name in subj:
                cam = subj[cam_name]
                new_desc['num_per_subj'][scen_name][subj_name] += cam 

    # get "num_per_scen"
    for scen_name in new_desc['num_per_subj']:
        scen = new_desc['num_per_subj'][scen_name]
        new_desc['num_per_scen'][scen_name] = 0
        for subj_name in scen:
            subj = scen[subj_name]
            new_desc['num_per_scen'][scen_name] += subj               
    return new_desc 

def get_sub_name(idx, dict_vals):
    for key, val in zip(dict_vals.keys(), dict_vals.values()):
        if idx < val:
            break
        idx -= val   
    return idx, key

def get_desc_f(desc):
    def f(idx):
        # compute scenario
        idx, scen = get_sub_name(idx, desc['num_per_scen'])
        # print(idx, scen, end='\n\n')

        # compute subject
        idx, subj = get_sub_name(idx, desc['num_per_subj'][scen])
        # print(idx, subj, end='\n\n')

        # compute camera
        idx, cam = get_sub_name(idx, desc['num_per_cam'][scen][subj])
        # print(idx, cam, end='\n\n')

        # compute index
        name = '{:06d}'.format(idx)
        return scen, subj, cam, name
    return f

def desc_validity_check(desc):
    print(desc['num_total'])
    print(sum([desc['num_per_scen'][scen] for scen in desc['num_per_scen']]))

    num_per_subj = 0
    for scen in desc['num_per_subj']:
        for subj in desc['num_per_subj'][scen]:
            num_per_subj += desc['num_per_subj'][scen][subj]
    print(num_per_subj)

    num_per_cam = 0
    for scen in desc['num_per_cam']:
        for subj in desc['num_per_cam'][scen]:
            for cam in desc['num_per_cam'][scen][subj]:
                num_per_cam += desc['num_per_cam'][scen][subj][cam]
    print(num_per_cam)

def gaussian(mu_x, mu_y, sig, size):
    xy = np.indices(size)
    x = xy[0,:,:]
    y = xy[1,:,:]

    psf  = np.exp(-((x-mu_x)**2/(2*sig**2) + (y-mu_y)**2/(2*sig**2)))
    return psf/psf.sum() # normalize or not?

def construct_heatmap(points_crop, points_visibility, sigma, map_width=56, orig_width=224, heatmaps_init=None, sum_channel=False):
    """Construct a heatmap, possibly starting from an existing stack (for multi-person) and adding an optional sum channel"""

    scale = map_width/orig_width
    points_crop = points_crop.copy()*scale
    numJoints = points_crop.shape[0]
    numChannels = numJoints
    if sum_channel:
        numChannels += 1
    if heatmaps_init is None:
        heatmaps = np.zeros((map_width, map_width, numChannels),dtype='float32')
    else:
        heatmaps = heatmaps_init
    for pi in range(0,numJoints):
        if points_visibility[pi]:
            heatmap = gaussian(points_crop[pi,1], points_crop[pi,0], sigma, heatmaps.shape[0:2])
            # heatmap = gaussian(points_crop[pi,1]*map_width, points_crop[pi,0]*map_width, sigma, heatmaps.shape[0:2])
            heatmaps[:,:,pi] = heatmap
            # heatmaps[:,:,pi] = np.maximum(heatmap, heatmaps[:,:,pi])
            
    if sum_channel:
        # combined heatmap for debugging purposes, also add as target, might help to supervise
        averageMap = np.sum(heatmaps[:,:,:numJoints], 2) # sum everything except the previous average
        averageMap = np.clip(averageMap, 0., 1.)
        heatmaps[:,:,numJoints] = averageMap
    return heatmaps

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

def prepare_input(img, joints, out_size=(224,224)):
    ''' Prepares data for ResNet50 architecture, 
    img_size = C x H x W.


    Joints must change correspondingly.

    img - tensor, C x H x W
    joints - tensor, N x 2 # for the case of 2d HPE
    out_size - default is (224,224) as ResNet50 requires
    '''

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
    0       -> 10; 
    1       -> 9; 
    2       -> 8; 
    3       -> 11; 
    4       -> 12; 
    5       -> 13;
    (6 + 7) -> 14;
    8       -> 1; 
    9       -> 0; 
    10      -> 4; 
    11      -> 3; 
    12      -> 2; 
    13      -> 5; 
    14      -> 6; 
    15      -> 7.
    '''
    joints_h36m = joints_mpii.clone()[:-1]
    joints_h36m[[10,9,8,11,12,13]] = joints_mpii[[0,1,2,3,4,5]]
    joints_h36m[[1,0,4,3,2,5,6,7]] = joints_mpii[[8,9,10,11,12,13,14,15]]
    joints_h36m[14] = (joints_mpii[6] + joints_mpii[7])/2
    return joints_h36m

# def get_rect(l,r,t,b):
#     return [[l,l,r,r,l], 
#             [b,t,t,b,b]]

def getitem(img_path, joints, use_heatmaps, map_width, orig_width):
    # image = th.tensor((imread(img_path)/255)).permute(2,0,1).float()

    image = Image.open(img_path)
    preprocess = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).float()

    joints_valid = joints.sum(dim=1) != 0 # 0,0 - non-visible points
    image, joints = prepare_input(image, joints, out_size=(orig_width,orig_width))

    if use_heatmaps:
        width = image.shape[-1] # image must be square!
        joints = construct_heatmap(joints.numpy(), 
                                   joints_valid, 
                                   sigma=SIGMA, 
                                   map_width=map_width, 
                                   orig_width=width)
        joints = th.tensor(joints).permute(2,0,1)
    return image, joints, joints_valid

def heatmap2joints(heatmap, orig_width=256):
    '''
    :param heatmap - tensor of size: B x num_joints x H x W
    
    Assume that H == W!
    '''
    if heatmap.ndim == 3:
        heatmap = heatmap.unsqueeze(0).clone()
    B, num_joints, H, W = heatmap.size()
    joints = th.zeros((B,num_joints,2))
    for b, h in enumerate(heatmap):
        for j in range(num_joints):
            argmax = th.argmax(h[j])
            joints[b, j, ...] = th.tensor([argmax % H,  argmax // H])
            
    return joints / H * orig_width

