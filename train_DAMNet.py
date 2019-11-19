import torch as th

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
# from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.nn import MSELoss as L2Loss, L1Loss, BCELoss

import models
from dataset_loader import MPII, H36M 


def run_training_damnet(datasets=(H36M, MPII), num_domains=2, batch_size=5, gpu_id=0, seed=0):

    th.manual_seed(seed)
    # device = th.device('cpu')
    device = th.device('cuda:{}'.format(gpu_id))

    feat_extr = models.DamNet(num_domains=num_domains).to(device)
    dom_discr = models.DomainClassifier(num_domains=num_domains).to(device)
    pose_regr = models.PoseRegressor().to(device)

    optimizer_feat_extr = Adam(feat_extr.parameters(), lr=1e-4)
    optimizer_dom_discr = Adam(dom_discr.parameters(), lr=1e-4)
    optimizer_pose_regr = Adam(pose_regr.parameters(), lr=1e-4)
    
    criterion_dom_discr = L1Loss()
    criterion_pose_regr = BCELoss()

    dataloaders = {}
    for i in range(num_domains):
        key = 'domain_'+str(i)
        dataloaders[key] = {}
        for num_images, mode, shuffle in zip([10000, 1000], ['train', 'val'], [True, False]):
            dataset = datasets[i](num_images=num_images, mode=mode)
            dataloaders[key][mode] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    print(dataloaders)

#######################################################################
#######################################################################
#######################################################################

if __name__ == '__main__':

    datasets = (H36M, MPII)
    num_domains = len(datasets)
    gpu_id = 2
    batch_size = 5

    run_training_damnet(datasets=datasets, num_domains=num_domains, 
        batch_size=batch_size, gpu_id=gpu_id)
    

    # # Runs the training...
    # training_procedure(start_epoch, opt.num_epochs, model, optimizer, scheduler, 
    #                     criterion, train_loader, val_loader, 
    #                     save_path, opt.save_each_epoch, opt.no_val)