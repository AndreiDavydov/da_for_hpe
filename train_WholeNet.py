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

def init_training(datasets=(H36M, MPII), num_domains=2, do_da=False,
        num_train_imgs=1000, num_val_imgs=100,
        batch_size=10):

    print(dt(), ' Loading model... ')
    model = models.WholeNet(num_domains, do_da=do_da).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

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
            dataloaders[i][mode] = DataLoader(dataset, batch_size=batch_size, 
                shuffle=shuffle, num_workers=5)
            print(dt(), ' Data for the domain {} (mode {}) is composed.'.format(i, mode))

    print(dt(), ' Initialization is done. \n________________________________')

    return model, optimizer, criterions, dataloaders


def compose_d(batch_size, domain_idx, num_domains):
    x = th.zeros((batch_size, num_domains))
    x[:,domain_idx] = 1
    return x.to(device)


def save_state(epoch, model, optimizer, losses):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    state = {'rng_state':th.get_rng_state(),\
             'rng_states_cuda':th.cuda.get_rng_state_all()}

    state['state_dict'] = model.state_dict()
    state['opt_state_dict'] = optimizer.state_dict()

    th.save(state, save_path+'state.pth')
    th.save(losses, save_path+'losses.pth')

    th.save(epoch, save_path+'epoch.pth')
    th.save(params, save_path+'params.pth')

    print("\n ===>", dt(), "epoch = {}. ".format(epoch))
    print("Model is saved to {}".format(save_path))


def load_state(model, optimizer):
    assert os.path.isdir(save_path), 'There is no the directory: {}'.format(save_path)

    state = th.load(save_path+'state.pth',
        map_location=lambda storage,loc:storage.to(device))
    
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['opt_state_dict'])  
    
    th.set_rng_state(state['rng_state'].cpu()) 

    losses = th.load(save_path+'losses.pth')
    params = th.load(save_path+'params.pth')
    epoch = th.load(save_path+'epoch.pth')

    return losses, params


def run_epoch(mode, model, optimizer, criterions, dataloaders, losses, epoch):
    hi_str = ' Training...' if mode == 'train' else ' Validation...'
    print(dt(), hi_str)

    print(dt(), 'lengths of dataloaders: ', 
        [len(dataloaders[domain][mode]) for domain in dataloaders])

    for batch_idx, (sample0, sample1) in tqdm(enumerate(zip(dataloaders[0][mode],\
                                        dataloaders[1][mode]))):
        # sample0 - H36M
        # sample1 - MPII

        # if batch_idx+1 % 10 == 0:
        #     print(dt(), 'Batch {}'.format(batch_idx))
        for domain_idx, sample in enumerate([sample0, sample1]):

            if mode == 'train':
                optimizer.zero_grad()

            imgs, joints, joints_valid = sample
            imgs = imgs.float().to(device)

            if mode == 'val':
                with th.no_grad(): 
                    p_hat, d_hat, z = model(imgs, domain_idx)
            else:
                p_hat, d_hat, z = model(imgs, domain_idx)

            if model.do_da:
                d = compose_d(d_hat.size(0), domain_idx, model.num_domains)
                d_loss = criterions['dd'](d_hat, d)
                losses[domain_idx]['dd'][mode].append(d_loss.data.cpu().numpy().item())

            if domain_idx == 0: # source domain, has labels
                p = joints.float().to(device)
                p_loss = criterions['pr'](p_hat[joints_valid], p[joints_valid])
                losses[domain_idx]['pr'][mode].append(p_loss.data.cpu().numpy().item())
                
            if mode == 'train':
                if domain_idx == 0:
                    p_loss.backward(retain_graph=True) if model.do_da else \
                    p_loss.backward(retain_graph=False)
                if model.do_da:
                    d_loss.backward()
                optimizer.step()
        

    #######################################################################
    #######################################################################
    #######################################################################

def main():

    opt = parse_args()

    datasets = (H36M, MPII)
    num_domains = len(datasets)
    gpu_id = opt.gpu_id
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
    save_each_epoch = opt.save_each_epoch
    num_train_imgs = opt.num_train_imgs 
    num_val_imgs = opt.num_val_imgs
    do_da = opt.do_da
    return_z = opt.return_z

    global seed
    seed = opt.seed

    global exp_name
    exp_name = opt.exp_name

    global save_path
    save_path = SAVE_PATH+exp_name+'/' if exp_name is not None else SAVE_PATH

    global device
    device = th.device('cuda:{}'.format(gpu_id)) # cpu is not even considered
    
    th.manual_seed(seed)

    global params
    params = {'datasets':datasets, 'num_domains':num_domains, 'gpu_id':gpu_id, 
        'batch_size':batch_size, 'num_epochs':num_epochs, 'save_each_epoch':save_each_epoch, 
        'num_train_imgs':num_train_imgs, 'num_val_imgs':num_val_imgs,
        'do_da':do_da, 'return_z':return_z, 'save_path':save_path}

    init = init_training(datasets, num_domains, do_da,
        num_train_imgs, num_val_imgs, batch_size)

    model, optimizer, criterions, dataloaders = init

    # only two domains: H36M and MPII
    # only two losses: dd and pr. 
    losses = {0:{'dd':{'train':[], 'val':[]}, 'pr':{'train':[], 'val':[]}},
              1:{'dd':{'train':[], 'val':[]}, 'pr':{'train':[], 'val':[]}}}

    for epoch in range(1, num_epochs+1):
        run_epoch('train', model, optimizer, criterions, dataloaders, losses, epoch)
        run_epoch('val', model, optimizer, criterions, dataloaders, losses, epoch)
        
        progress = epoch/num_epochs
        model.update_plasts(progress)
        model.update_lambd(progress)

        tmp_losses = losses.copy()
        th.save(tmp_losses, save_path+'tmp_losses.pth')

        if epoch % save_each_epoch == 0 or epoch == 1:
            save_state(epoch, model, optimizer, losses)

        print('----------------------------------------')
        print(dt(), 'epoch {} out of {} is finished.'.format(epoch, num_epochs))
        print('----------------------------------------')
    

if __name__ == '__main__':

    main()

    