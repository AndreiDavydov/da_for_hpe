import torch as th

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import MSELoss as L2Loss, L1Loss, BCELoss

from tqdm import tqdm
from datetime import datetime
import os

import models
from datasets import MPII, H36M 
from arg_parser import parse_args



SAVEPATH = './saved_models/'
DATASETS = (H36M, MPII)[::-1]

LR0 = 1e-3
NUMWORKERS = 20
OPTIM = Adam

SEED = 0

def dt():
    return datetime.now().strftime('%H:%M:%S')

def init_training(params):

    if not os.path.isdir(params['save_path']):
        os.mkdir(params['save_path'])

    if th.cuda.is_available():
        device = 'cuda:{}'.format(params['gpu_id'])
    else:
        print('Warning: You tried to use CUDA GPU, but it is not available. CPU is used for training.\n')
        device = 'cpu'

    device = th.device(device) 
    params['device'] = device

    print(dt(), ' Loading model... ')

    model = models.WholeNet(params['num_domains'], 
                            params['pretrained'], 
                            params['z_features'], 
                            params['hidden_features'], 
                            do_da=params['do_da'], 
                            use_heatmaps=params['use_heatmaps'], 
                            return_z=params['return_z']).to(device)

    optimizer = OPTIM(model.parameters(), lr=LR0)

    criterion_dom_discr = BCELoss()
    criterion_pose_regr = L1Loss() if params['criterion'] == 'L1' else L2Loss()
    criterions = {'dd':criterion_dom_discr, 'pr':criterion_pose_regr}

    print(dt(), ' Preparing data loaders... ')

    dataloaders = {}
    for i in range(params['num_domains']):
        dataloaders[i] = {}
        for num_images, mode, shuffle in zip(
                                             [params['num_train_imgs'], params['num_val_imgs']], 
                                             ['train', 'val'], 
                                             [True, False]):
            if mode == 'val' and not params['do_val']:
                continue
            dataset = params['datasets'][i](num_images, mode, params['use_heatmaps'])
            params['num_'+mode+'_imgs'] = len(dataset)
            dataloaders[i][mode] = DataLoader(dataset, batch_size=params['batch_size'], 
                                                    shuffle=shuffle, 
                                                    num_workers=NUMWORKERS)

            print(dt(), ' Data for the domain {} (mode {}) is composed.'.format(i, mode))

    print(dt(), ' Initialization is done. \n________________________________')

    return model, optimizer, criterions, dataloaders


def compose_d(batch_size, domain_idx, num_domains):
    x = th.zeros((batch_size, num_domains))
    x[:,domain_idx] = 1
    return x


def save_state(epoch, model, optimizer, losses, params):

    state = {'rng_state':th.get_rng_state(),\
             'rng_states_cuda':th.cuda.get_rng_state_all()}

    state['state_dict'] = model.state_dict()
    state['opt_state_dict'] = optimizer.state_dict()

    print('save, ', epoch)
    if epoch % 5 == 0:
        th.save(state, params['save_path']+'state_'+'{:04d}'.format(epoch)+'.pth')
        print('MODEL PARAMETERS ARE SAVED TO:')
        print(params['save_path']+'state_'+'{:04d}'.format(epoch)+'.pth')
    th.save(losses, params['save_path']+'losses.pth')

    th.save(epoch, params['save_path']+'epoch.pth')
    th.save(params, params['save_path']+'params.pth')

    print("\n ===>", dt(), "epoch = {}. ".format(epoch))
    print("Model is saved to {}".format(params['save_path']))


# def load_state(model, optimizer, save_path, device):
#     assert os.path.isdir(save_path), 'There is no the directory: {}'.format(save_path)

#     state = th.load(save_path+'state.pth',
#         map_location=lambda storage,loc:storage.to(device))
    
#     model.load_state_dict(state['state_dict'])
#     optimizer.load_state_dict(state['opt_state_dict'])  
    
#     th.set_rng_state(state['rng_state'].cpu()) 

#     losses = th.load(save_path+'losses.pth')
#     params = th.load(save_path+'params.pth')
#     epoch = th.load(save_path+'epoch.pth')

#     return losses, params


def run_epoch(mode, model, optimizer, criterions, dataloaders, losses, params):
    hi_str = ' Training...' if mode == 'train' else ' Validation...'
    print(dt(), hi_str)

    for domain_idx in dataloaders: # decided to do whole epoch for one (by one) domain
        if len(dataloaders) > 1:
            print('domain ', domain_idx) 
        length_dl = len(dataloaders[domain_idx][mode]) 
        for batch_idx, sample in tqdm(enumerate(dataloaders[domain_idx][mode]), desc='batch idx (out of {})'.format(length_dl)):

            if mode == 'train':
                optimizer.zero_grad()

            imgs, joints, joints_valid = sample
            imgs = imgs.to(params['device'])

            if mode == 'val':
                model.train()
                with th.no_grad(): p_hat, d_hat, z = model(imgs, domain_idx)
            else:
                model.eval()
                p_hat, d_hat, z = model(imgs, domain_idx)

            if model.do_da:
                d = compose_d(d_hat.size(0), domain_idx, model.num_domains).to(params['device'])
                d_loss = criterions['dd'](d_hat, d)
                losses[domain_idx]['dd'][mode].append(d_loss.data.cpu().numpy().item())

            if domain_idx == 0: # source domain, has labels
                p = joints.to(params['device'])
                p_loss = criterions['pr'](p_hat[joints_valid], p[joints_valid])

                losses[domain_idx]['pr'][mode].append(p_loss.data.cpu().numpy().item())
                
            if mode == 'train':
                if domain_idx == 0:
                    p_loss.backward(retain_graph=True) if model.do_da else p_loss.backward(retain_graph=False)
                if model.do_da:
                    d_loss.backward()
                optimizer.step()
        

    #######################################################################
    #######################################################################
    #######################################################################

def print_params(params):
    print('\n____________________________________\n')
    print('parameters of the current experiment: \n')
    for key in params:
        if key == 'datasets':
            print("{0:<20}{1:>20}".format(key, str(params[key])))
            continue
        print("{0:<20}{1:>20}".format(key, params[key])) 
    print('____________________________________\n\n')

def parse_arguments():
    # first, parse arguments from the input parameters.
    opt = parse_args()

    exp_name = opt.exp_name
    save_path = SAVEPATH+exp_name+'/' if exp_name is not None else SAVEPATH

    params = {'batch_size':             opt.batch_size, 
              'criterion':              opt.loss,
              'num_epochs':             opt.num_epochs, 
              'save_each_epoch':        opt.save_each_epoch, 
              'z_features':             opt.z_features,  
              'hidden_features':        opt.hidden_features, 
              'pretrained':             not opt.no_pretrained,
              'gpu_id':                 opt.gpu_id,
              'verbose':                not opt.no_verbose,
              'num_train_imgs':         opt.num_train_imgs,
              'num_val_imgs':           opt.num_val_imgs,
              'do_da':                  not opt.no_da,
              'do_val':                 not opt.no_val,      
              'return_z':               opt.return_z,
              'one_flow':              opt.one_flow,
              'use_heatmaps':           not opt.map_joints,
              'save_path':              save_path
              }

    params['datasets'] = (DATASETS[0],) if params['one_flow'] else DATASETS
    params['num_domains'] = len(params['datasets'])

    print_params(params)

    return params

def main():

    params = parse_arguments()
    
    init = init_training(params)
    model, optimizer, criterions, dataloaders = init

    # only two domains: H36M and MPII
    # only two losses: dd and pr. 
    losses = {0:{'dd':{'train':[], 'val':[]}, 'pr':{'train':[], 'val':[]}},
              1:{'dd':{'train':[], 'val':[]}, 'pr':{'train':[], 'val':[]}}}

    # let's add scheduler:
    scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, [50,100,150,200], gamma=0.2, last_epoch=-1)

    for epoch in range(1, params['num_epochs']+1):
        run_epoch('train', model, optimizer, criterions, dataloaders, losses, params)
        if params['do_val']:
            run_epoch('val', model, optimizer, criterions, dataloaders, losses, params)
        
        progress = epoch/params['num_epochs']
        if not params['one_flow']:
            model.update_plasts(progress)
        if model.do_da:
            model.update_lambd(progress)
        
        if scheduler is not None:
            scheduler.step()

        if epoch % params['save_each_epoch'] == 0 or epoch == 1:
            save_state(epoch, model, optimizer, losses, params)

        print('----------------------------------------')
        print(dt(), 'epoch {} out of {} is finished.'.format(epoch, params['num_epochs']))
        print('----------------------------------------')
    

if __name__ == '__main__':

    main()

    