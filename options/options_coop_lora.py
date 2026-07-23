import numpy as np
import os
import glob
import torch
import argparse


def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('-dataset', default='multi', help='miniImagenet/CropDisease/EuroSAT/ISIC2018/ChestX, specify multi for training with multiple domains')
    parser.add_argument('-test_n_way', default=5, type=int,    help='class num to classify for testing (validation) ')
    parser.add_argument('-n_shot', default=5, type=int,    help='number of labeled data in each class, same as n_support')
    parser.add_argument('-aug', action='store_true',  help='perform data augmentation or not during training ')

    parser.add_argument('-split', default='novel', help='base/val/novel')
    parser.add_argument('-ckp_path', default=None, help='source-domain-trained checkpoint')
    
    ### clip_lora ###
    parser.add_argument('-data_path', default='/home/yxz/cdfsl_dataset', type=str, help='')
    parser.add_argument('-save_path', default='./output/CLIP-LoRA', type=str, help='')
    parser.add_argument('-backbone', default='ViT-B/16', type=str)
    parser.add_argument('-encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('-position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'])
    parser.add_argument('-params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA')
    parser.add_argument('-r', default=16, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('-alpha', default=8, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('-dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module') 
    
    parser.add_argument('-base_lr', default=2e-4, type=float)
    parser.add_argument('-special_lr', default=2e-4, type=float)     
    parser.add_argument('-lora_lr', default=2e-4, type=float)
    parser.add_argument('-coop_lr', default=2e-3, type=float)
    parser.add_argument('-seed', default=1, type=int)
    parser.add_argument('-total_epoch', default=100, type=int)
    parser.add_argument('-save_lora', default=False, type=bool)
    parser.add_argument('-filename', default="clip-lora", type=str)
    parser.add_argument('-lora_type', type=str, default='linear')
    # parser.add_argument('-ctx_init', default="a photo of a ", type=str)

    parser.add_argument('-aux_param1', default=0.0, type=float)   #7
    parser.add_argument('-aux_param2', default=0.0, type=float)  #8
    parser.add_argument('-pos_rate', default=1.0, type=float)  #8  
    parser.add_argument('-aux_container', default={}, type=dict) 
    parser.add_argument('-beta_a', type=float, default=0.0)
    parser.add_argument('-beta_b', type=float, default=0.0)
    parser.add_argument('-loss_rate', default=0.0, type=float) 

    parser.add_argument('-droplayer', type=int, default=13)
    parser.add_argument('-add_ratio', type=float, default=0.0)
    parser.add_argument('-minus_ratio', type=float, default=0.0)
    parser.add_argument('-temper', type=float, default=0.1)
    parser.add_argument('-use_all_texts', default=False, type=bool)
    parser.add_argument('-suppress_positions', default=False, type=bool)
    parser.add_argument('-enhance_positions', default=False, type=bool)
    parser.add_argument('-txtlayer', default=10, type=int)
    parser.add_argument('-txtrate', default=1, type=float)
    parser.add_argument('-minus_txtrate', default=1, type=float)
    parser.add_argument('-txtlinearmin', default=0.0, type=float)
    parser.add_argument('-txt_temperature', default=1.0, type=float)
    parser.add_argument('-interval_size_ratio', default=0.1, type=float)
    parser.add_argument('-decay_rate', default=0.1, type=float)
    parser.add_argument('-method', default="none",  type=str)
    parser.add_argument('-use_pacl', default=False, type=bool)
    parser.add_argument('-training', default=False, type=bool)
    parser.add_argument('-distancetype', default="cosine",  type=str)

    parser.add_argument('-test_mode', default=False, type=bool)
    parser.add_argument('-current_episode', default=0, type=int)
    parser.add_argument('-dataset_name', default="none", type=str)

    parser.add_argument('-visualize_attention', default=False, type=bool)
    parser.add_argument('-max_heatmap_episodes', default=0, type=int)
    parser.add_argument('-max_images_per_episode', default=0, type=int)
    parser.add_argument('-collect_statistics', default=False, type=str)
    
  
   
    # ##### coop #######
    # parser.add_argument('-model_name', type=str, default="ViT-B/16")
    parser.add_argument('-n_ctx', type=int, default=4)
    parser.add_argument('-ctx_init', type=str, default="a photo of a")
    parser.add_argument('-img_size', type=int, default=224)
    # parser.add_argument('-layer', type=int, default=0)
    # parser.add_argument('-csc', type=bool, default=False, help="class-specific context")
    parser.add_argument('-ctp', type=str, default="end", help="class_token_position")
    # parser.add_argument('-init_weights', type=str, default="")
    # parser.add_argument('-use_cuda', type=bool, default=True)
    # # parser.add_argument('-aug', type=bool, default=False)

    parser.add_argument('-test_bs', default=64, type=int, help ='test batch size')
    parser.add_argument('-visual_attn_drop_rate', default=0.0, type=float)#0.2
    parser.add_argument('-text_attn_drop_rate', default=0.0, type=float)#0.1
    

    return parser.parse_args()


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir, resume_epoch=-1):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    epoch = max_epoch if resume_epoch == -1 else resume_epoch
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
    return resume_file

def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

def load_warmup_state(filename, method):
    print('  load pre-trained model file: {}'.format(filename))
    warmup_resume_file = get_resume_file(filename)
    print(' warmup_resume_file:', warmup_resume_file)
    tmp = torch.load(warmup_resume_file)
    if tmp is not None:
        state = tmp['state']
        state_keys = list(state.keys())
        print(state_keys)
        for i, key in enumerate(state_keys):
            if 'relationnet' in method and "feature." in key:
                newkey = key.replace("feature.","")
                state[newkey] = state.pop(key)
            elif method == 'gnnnet' and 'feature.' in key:
                newkey = key.replace("feature.","")
                state[newkey] = state.pop(key)
            elif method == 'matchingnet' and 'feature.' in key and '.7.' not in key:
                newkey = key.replace("feature.","")
                state[newkey] = state.pop(key)
            elif('VAE_model.' in key):
                newkey = key.replace("VAE_model.","")
                state[newkey] = state.pop(key)

            else:
                state.pop(key)
    else:
        raise ValueError(' No pre-trained encoder file found!')
    return state



def load_warmup_state_speci(filename, method):
    print('  load pre-trained model file: {}'.format(filename))
    #warmup_resume_file = get_resume_file(filename)
    warmup_resume_file = filename
    print(' warmup_resume_file:', warmup_resume_file)
    tmp = torch.load(warmup_resume_file)
    if tmp is not None:
        state = tmp['state']
        state_keys = list(state.keys())
        print(state_keys)
        for i, key in enumerate(state_keys):
            if 'relationnet' in method and "feature." in key:
                newkey = key.replace("feature.","")
                state[newkey] = state.pop(key)
            elif method == 'gnnnet' and 'feature.' in key:
                newkey = key.replace("feature.","")
                state[newkey] = state.pop(key)
            elif method == 'matchingnet' and 'feature.' in key and '.7.' not in key:
                newkey = key.replace("feature.","")
                state[newkey] = state.pop(key)
            elif('VAE_model.' in key):
                newkey = key.replace("VAE_model.","")
                state[newkey] = state.pop(key)
            else:
                state.pop(key)
    else:
        raise ValueError(' No pre-trained encoder file found!')
    return state
