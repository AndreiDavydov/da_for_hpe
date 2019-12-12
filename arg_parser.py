'''
This module provides Parser description, as it can be executed from terminal.
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training of DAMNet.')
    parser.add_argument('--no_verbose', action='store_true', help='Whether NOT to show TQDM iterations interface... '+\
                                                                     '(False by default)')
    parser.add_argument('--loss', type=str, default='L1', help='shortage for Criterion function ("L1" by default) ')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--save_each_epoch', type=int, default=10, help='With what period (in epochs) to save models')
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs to train for.')
    parser.add_argument('--num_train_imgs', type=int, default=1280, help='# of images to train on.')
    parser.add_argument('--num_val_imgs', type=int, default=1280, help='# of images to validate on.')
    parser.add_argument('--no_pretrained', action='store_true', help='Use network without pretraining (True) or with it. '+\
                                                                            '(False by default)')
    parser.add_argument('--exp_name', type=str, default='TEST_RUN', help='model short type name, additional folder division. ')
    parser.add_argument('--gpu_id', type=int, default=0, help='which GPU to use?')
    parser.add_argument('--no_da', action='store_true', help='not to do Domain Adaptation? '+\
                'If True then additional DA network will NOT be initialized. (False by default)')
    parser.add_argument('--return_z', action='store_true', help='return latent vector Z vector? '+\
                'Usually it is used for debug purposes (False by default)')
    parser.add_argument('--no_val', action='store_true', help='Whether NOT to do validation. '+\
                                '(False by default)')
    parser.add_argument('--one_flow', action='store_true', help='Whether to train only one flow. Gates are deactivated.'+\
                                '(False by default)')
    parser.add_argument('--map_joints', action='store_true', help='Whether to train on Joints or Heatmaps. Heatmaps as default.')
    parser.add_argument('--z_features', type=int, default=64, help='the dimension of the feature space '+\
                                                                    '(right after Feature Extractor).')
    parser.add_argument('--hidden_features', type=int, default=64, help='the dimension of the hidden linear layers '+\
                                                                        'in the Pose Regressor network.')

    opt = parser.parse_args()
    return opt 