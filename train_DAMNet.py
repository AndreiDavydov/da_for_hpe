import torch as th

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import MSELoss as L2Loss, L1Loss, BCELoss

from tqdm import tqdm
from datetime import datetime
import os

import models
from dataset_loader import MPII, H36M 
from arg_parser import parse_args



SAVE_PATH = './saved_models/'

def dt():
    return datetime.now().strftime('%H:%M:%S')

def init_training(datasets=(H36M, MPII), num_domains=2, 
        num_train_imgs=1000, num_val_imgs=100,
        batch_size=10, seed=0):

    th.manual_seed(seed)

    print(dt(), ' Loading models... ')
    feat_extr = models.DamNet(num_domains=num_domains).to(device)
    dom_discr = models.DomainClassifier(num_domains=num_domains).to(device)
    pose_regr = models.PoseRegressor().to(device)
    sub_nets = {'fe':feat_extr, 'dd':dom_discr, 'pr':pose_regr}

    optimizer_feat_extr = Adam(feat_extr.parameters(), lr=1e-4)
    optimizer_dom_discr = Adam(dom_discr.parameters(), lr=1e-4)
    optimizer_pose_regr = Adam(pose_regr.parameters(), lr=1e-4)
    optimizers = {'fe':optimizer_feat_extr, 'dd':optimizer_dom_discr, 'pr':optimizer_pose_regr}
    
    criterion_dom_discr = BCELoss()
    criterion_pose_regr = L1Loss()
    criterions = {'dd':criterion_dom_discr, 'pr':criterion_pose_regr}

    print(dt(), ' Preparing data loaders... ')

    dataloaders = {}
    for i in range(num_domains):
        dataloaders[i] = {}
        for num_images, mode, shuffle in zip([num_train_imgs, num_val_imgs], 
                                             ['train', 'val'], 
                                             [True, False]):
            dataset = datasets[i](num_images=num_images, mode=mode)
            dataloaders[i][mode] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    print(dt(), ' Initialization is done. ')

    return sub_nets, optimizers, criterions, dataloaders


def compose_d(batch_size, domain_idx):
    x = th.zeros((batch_size,2))
    x[:,domain_idx] = 1
    return x.to(device)


def save_state(epoch, sub_nets, optimizers, losses):
    save_path = SAVE_PATH+exp_name+'/' if exp_name is not None else SAVE_PATH
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    state = {'rng_state':th.get_rng_state(),\
             'rng_states_cuda':th.cuda.get_rng_state_all()}

    for key in ['fe', 'dd', 'pr']:
        state[key+'_state_dict'] = sub_nets[key].state_dict()
        state[key+'_opt_state_dict'] = optimizers[key].state_dict()

    th.save(state, save_path+'state.pth')
    th.save(losses, save_path+'losses.pth')
    th.save({'epoch':epoch}, save_path+'params.pth')

    print("\n ===>", dt(), "epoch = {}. ".format(epoch))
    print("Model is saved to {}".format(save_path))


def load_state(sub_nets, optimizers):
    save_path = SAVE_PATH+exp_name+'/' if exp_name is not None else SAVE_PATH
    assert os.path.isdir(save_path), 'There is no the directory: {}'.format(save_path)

    state = th.load(save_path+'state.pth',
        map_location=lambda storage,loc:storage.to(device))
    
    for key in ['fe', 'dd', 'pr']:
        sub_nets[key].load_state_dict(state[key+'_state_dict'])
        optimizers[key].load_state_dict(state[key+'_opt_state_dict'])  
    
    th.set_rng_state(state['rng_state'].cpu()) 

    losses = th.load(save_path+'losses.pth')
    params = th.load(save_path+'params.pth')

    return losses, params


def run_epoch(mode, sub_nets, optimizers, criterions, dataloaders, losses):
    hi_str = ' Training...' if mode == 'train' else ' Validation...'
    print(dt(), hi_str)

    print(dt(), 'lengths of dataloaders: ', 
        [len(dataloaders[domain][mode]) for domain in dataloaders])

    for sample0, sample1 in tqdm(zip(dataloaders[0][mode],\
                                    dataloaders[1][mode])):
        # sample0 - H36M
        # sample1 - MPII

        for domain_idx, sample in enumerate([sample0, sample1]):

            if mode == 'train':
                [opt.zero_grad() for opt in optimizers.values()]

            imgs, joints, joints_valid = sample
            imgs = imgs.float().to(device)

            if mode == 'val':
                with th.no_grad(): 
                    z = sub_nets['fe'](imgs, domain_idx)
                    d_hat = sub_nets['dd'](z)
            else:
                z = sub_nets['fe'](imgs, domain_idx)
                d_hat = sub_nets['dd'](z)

            d = compose_d(d_hat.size(0), domain_idx)
            dd_loss = criterions['dd'](d_hat, d)

            losses[domain_idx]['dd'][mode].append(dd_loss.data.cpu().numpy().item())

            if domain_idx == 0: # source domain, has labels
                p_hat = joints.float().to(device)
                if mode == 'val':
                    with th.no_grad(): p = sub_nets['pr'](z)
                else:
                   p = sub_nets['pr'](z) 
                
                p_loss = criterions['pr'](p_hat[joints_valid], p[joints_valid])
                losses[domain_idx]['pr'][mode].append(p_loss.data.cpu().numpy().item())
                
                if mode == 'train':
                    p_loss.backward(retain_graph=True)
                    optimizers['pr'].step()

            if mode == 'train':
                dd_loss.backward()
                optimizers['fe'].step()
                optimizers['dd'].step()          

    #######################################################################
    #######################################################################
    #######################################################################

if __name__ == '__main__':

    opt = parse_args()

    datasets = (H36M, MPII)
    num_domains = len(datasets)
    gpu_id = opt.gpu_id
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
    save_each_epoch = opt.save_each_epoch
    num_train_imgs = opt.num_train_imgs 
    num_val_imgs = opt.num_val_imgs

    global exp_name
    exp_name = opt.exp_name

    global device
    device = th.device('cuda:{}'.format(gpu_id))

    init = init_training(datasets=datasets, 
        num_domains=num_domains, batch_size=batch_size, 
        num_train_imgs=num_train_imgs, num_val_imgs=num_val_imgs)
    sub_nets, optimizers, criterions, dataloaders = init

    # only two domains: H36M and MPII
    # only two losses: dd and pr. 
    losses = {0:{'dd':{'train':[], 'val':[]}, 'pr':{'train':[], 'val':[]}},
              1:{'dd':{'train':[], 'val':[]}, 'pr':{'train':[], 'val':[]}}}

    for epoch in range(1, num_epochs+1):
        run_epoch('train', sub_nets, optimizers, criterions, dataloaders, losses)
        run_epoch('val', sub_nets, optimizers, criterions, dataloaders, losses)

        if epoch % save_each_epoch == 0 or epoch == 1:
            save_state(epoch, sub_nets, optimizers, losses)

        print('----------------------------------------')
        print(dt(), 'epoch {} out of {} is finished.'.format(epoch, num_epochs))
        print('----------------------------------------')
    




