'''
This module provides Parser description, as it can be executed from terminal.
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training of DAMNet.')

    parser.add_argument('--batch_size', type=int, default=10, help='batch size.')
    parser.add_argument('--seed', type=int, default=0, help='Seed number.')
    parser.add_argument('--save_each_epoch', type=int, default=10, help='With what period (in epochs) to save models')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for.')
    parser.add_argument('--num_train_imgs', type=int, default=100, help='# of images to train on.')
    parser.add_argument('--num_val_imgs', type=int, default=100, help='# of images to validate on.')
    parser.add_argument('--exp_name', type=str, default=None, help='model short type name, additional folder division. ')
    parser.add_argument('--gpu_id', type=int, default=0, help='which GPU to use?')
    parser.add_argument('--do_da', action='store_true', help='do Domain Adaptation? '+\
                'Then additional network will be initialized. (False by default)')
    parser.add_argument('--return_z', action='store_true', help='return latent vector Z vector? '+\
                'Usually it is used for debug purposes (False by default)')
    
    opt = parser.parse_args()
    return opt 