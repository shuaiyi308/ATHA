import argparse


def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('-dataset', default='multi', help='miniImagenet/CropDisease/EuroSAT/ISIC2018/ChestX, specify multi for training with multiple domains')
    parser.add_argument('-test_n_way', default=5, type=int,    help='class num to classify for testing (validation) ')
    parser.add_argument('-n_shot', default=5, type=int,    help='number of labeled data in each class, same as n_support')
    parser.add_argument('-aug', action='store_true',  help='perform data augmentation or not during training ')
    parser.add_argument('-data_path', default='/home/yxz/cdfsl_dataset', type=str, help='')

    parser.add_argument('-split', default='novel', help='base/val/novel')
    
    ##### coop #######
    parser.add_argument('-model_name', type=str, default="ViT-B/16")
    parser.add_argument('-n_ctx', type=int, default=4)
    parser.add_argument('-ctx_init', type=str, default="a photo of a")
    parser.add_argument('-img_size', type=int, default=224)
    parser.add_argument('-csc', type=bool, default=False, help="class-specific context")
    parser.add_argument('-ctp', type=str, default="end", help="class_token_position")
    parser.add_argument('-init_weights', type=str, default="")
    parser.add_argument('-use_cuda', type=bool, default=True)

    parser.add_argument('-aux_param', default=0.0, type=float)
    parser.add_argument('-aux_param2', default=0.0, type=float)
    parser.add_argument('-aux_container', default={}, type=dict)
    
 
    parser.add_argument('-loss_rate', default=0.0, type=float) 

    parser.add_argument('-droplayer', type=int, default=13)
    parser.add_argument('-mask_ratio', type=float, default=0.0)
    parser.add_argument('-mask_ratio_small', type=float, default=0.0)
    parser.add_argument('-temper_rate', type=float, default=0.5)
    parser.add_argument('-suppress_positions', default=False, type=bool)
    parser.add_argument('-enhance_positions', default=False, type=bool)
    
    #### coop_optimizer ####
    parser.add_argument('-optim_name', type=str, default="sgd")
    parser.add_argument('-optim_lr', type=float, default=0.002)
    parser.add_argument('-optim_wegiht_decay', type=float, default=5e-4)
    parser.add_argument('-optim_momentum', type=float, default=0.9)
    parser.add_argument('-optim_sdg_dampning', type=int, default=0)
    parser.add_argument('-optim_sdg_nesterov', type=bool, default=False)
    parser.add_argument('-optim_rmsprop_alpha', type=float, default=0.99)
    parser.add_argument('-optim_adam_beta1', type=float, default=0.9)
    parser.add_argument('-optim_adam_beta2', type=float, default=0.999)
    parser.add_argument('-optim_staged_lr', type=bool, default=False)
    parser.add_argument('-optim_new_layers', type=tuple, default=())
    parser.add_argument('-optim_base_lr_mult', type=float, default=0.1)
    
    parser.add_argument('-optim_max_epoch', type=int, default=50)
    parser.add_argument('-optim_lr_scheduler', type=str, default="cosine")
    parser.add_argument('-optim_stepsize', type=tuple, default=(-1, ))
    parser.add_argument('-optim_gamma', type=float, default=0.1)
    
    parser.add_argument('-optim_warmup_epoch', type=int, default=1)
    parser.add_argument('-optim_warmup_recount', type=bool, default=True)
    parser.add_argument('-optim_warmup_type', type=str, default="constant")
    parser.add_argument('-optim_warmup_cons_lr', type=float, default=1e-5)
    parser.add_argument('-optim_warmup_min_lr', type=float, default=1e-5)
    
    parser.add_argument('-output_dir', type=str, default="./output")
    return parser.parse_args()
