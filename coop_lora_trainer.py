import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import random
import os

import torch.optim
from itertools import combinations
import h5py
from data.datamgr import SetDataManager # use this line if aug is not used
#from data.datamgr_aug import SetDataManager # use this line if aug
from options.options_coop_lora import parse_args
from find_name_novel import return_novel_label_names
from utils import *
from lora_utils import *
import torch.nn.functional as F

from datetime import datetime

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from loralib.utils import apply_lora, get_lora_parameters

import os

#import time
#time.sleep(5400)


_tokenizer = _Tokenizer()

def load_clip_to_cpu(args):
    backbone_name = args.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class TextEncoder(nn.Module): 
    def __init__(self, clip_model, params):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype


    def forward(self, prompts, tokenized_prompts, params=None):

        params.aux_container['txt_pos'] = self.positional_embedding.type(self.dtype).unsqueeze(0).permute(1, 0, 2)
        params.aux_container['tokenized_prompts_argmax'] = tokenized_prompts.argmax(dim=-1)


    
        x = prompts +  self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x, params=params)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)


        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
      
        xs = []
    
        
        
        
        for i in range(12):
            xs.append(self.ln_final(params.aux_container['text_block_output_%d'%i].permute(1, 0, 2)).type(self.dtype)[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection)

            
        params.aux_container['txt_block_feats'] = xs

        return x




class PromptLearner(nn.Module):  # prompt learner
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames) 
        n_ctx = args.n_ctx  # context 
        ctx_init = args.ctx_init # context initialization
        # dtype = clip_model.dtype
        dtype = torch.float16
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = args.img_size
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init: 
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else: 
            # random initialization
            if args.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)


        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args.ctp
        
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        # print(prompts.grad)
        
        return prompts

class CustomCLIP(nn.Module):  # CLIP
    def __init__(self, args, classnames, clip_model, params):
        super().__init__()
        self.prompt_learner = PromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, params)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype


    def forward(self, image, label=None, params=None, sp_img=None):
        # image_features: [b, c]; text_features: [num_class, c]; label: [b]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            prompts = self.prompt_learner()        
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts, params=params) # [num_block, num_class, c]
            params.aux_container['txt_feat_final'] = text_features
            params.aux_container['txt_feat'] = params.aux_container['txt_block_feats'][params.txtlayer]

            image_features = self.image_encoder(image, params)
            

            if sp_img is not None:
                sp_img_feats = self.image_encoder(sp_img, params) # [B, c]
                if params.aug:
                    sp_img_feats = sp_img_feats.reshape([-1, params.test_n_way, params.n_shot, sp_img_feats.shape[-1]])
                    sp_img_feats = sp_img_feats.mean(dim=[0,2]) # [cps, c]
                else:
                    sp_img_feats = sp_img_feats.reshape([params.test_n_way, params.n_shot, sp_img_feats.shape[-1]])
                    sp_img_feats = sp_img_feats.mean(dim=1) # [cps, c]
                sp_img_feats = F.normalize(sp_img_feats, dim=-1).half()


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = 4 if params.n_shot == 5 else 2

        logits = logit_scale * image_features @ text_features.t()

        if sp_img is not None:
            logits = logits + params.aux_param * image_features @ sp_img_feats.t() * logit_scale

        return logits

def save_features(features, labels, episode_idx, dataset_name, feature_type, feature_dir="features"):
    os.makedirs(feature_dir, exist_ok=True)
    
    if dataset_name=='miniImageNet':
        dataset_dir = os.path.join(feature_dir,'ours_testonlyenhance' ,params.dataset,dataset_name)
    else:
        dataset_dir = os.path.join(feature_dir,'ours_testonlyenhance' ,dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    filename = f"{dataset_dir}/{feature_type}_episode_{episode_idx:04d}.h5"
    
    with h5py.File(filename, 'w') as f:
       
        f.create_dataset('features', data=features.cpu().numpy())
     
        f.create_dataset('labels', data=labels.cpu().numpy())
      
        f.attrs['episode_idx'] = episode_idx
        f.attrs['dataset_name'] = dataset_name
        f.attrs['feature_type'] = feature_type
        f.attrs['timestamp'] = datetime.now().isoformat()
        f.attrs['feature_shape'] = features.shape
        
    print(f"Saved {feature_type} features to {filename}")

def extract_and_save_features(model, images, labels, episode_idx, dataset_name, params, feature_type="image"):

    model.eval()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            if feature_type == "image":

                if images is not None and images.dtype != model.dtype:
                    images = images.to(model.dtype)
       
                features = model.image_encoder(images.cuda(), params=params)
            else:  # text features
          
                    prompts = model.prompt_learner()
                    tokenized_prompts = model.tokenized_prompts
                    features = model.text_encoder(prompts, tokenized_prompts, params=params)
       
            save_features(features, labels, episode_idx, dataset_name, feature_type)
        
    return features

def initialize_miniimagenet_loader(params):

    image_size = 224
    n_way = params.test_n_way
    n_support = params.n_shot
    n_query = 15
    

    mini_loadfile = os.path.join(params.data_path, "miniImagenet", params.split + '.json')
    
    if os.path.exists(mini_loadfile):
        mini_datamgr = SetDataManager(image_size, n_way, n_support, n_query, n_eposide=400)
        mini_loader = mini_datamgr.get_data_loader(mini_loadfile, aug=params.aug)
        return mini_loader, return_novel_label_names()
    else:
        print(f"Warning: miniImagenet data file {mini_loadfile} not found")
        return None, None

def process_miniimagenet_batch(model, mini_loader, episode_idx, params):

    n_support = params.n_shot
    n_query = 15
    if mini_loader is None:
        return
        
    try:
  
        mini_iter = iter(mini_loader)
        mini_x, mini_y = next(mini_iter)
        
        if params.aug:
            mini_idxs = [row[0].item() for row in mini_y[0]]
            mini_batch_classnames = [return_novel_label_names()[idx] for idx in mini_idxs]
            
       
            mini_sup_x = []
            mini_qry_x = []
            mini_label = []
            for x in mini_x:
                x = x.cuda()
                x_var = Variable(x)
                y_a_i = Variable(torch.from_numpy(np.repeat(range(params.test_n_way), params.n_shot))).cuda()
                x_b_i = x_var[:, n_support:,:,:,:].contiguous().view(params.test_n_way * 15, *x.size()[2:])
                x_a_i = x_var[:,:n_support,:,:,:].contiguous().view(params.test_n_way * params.n_shot, *x.size()[2:])
                mini_sup_x.append(x_a_i)
                mini_qry_x.append(x_b_i)
                mini_label.append(y_a_i)
            
            mini_x_a_i = torch.cat(mini_sup_x, dim=0)
            mini_y_a_i = torch.cat(mini_label, dim=0)
            mini_x_b_i = mini_qry_x[0]
        else:
            mini_x = mini_x.cuda()
            mini_x_var = Variable(mini_x)
            mini_y_a_i = Variable(torch.from_numpy(np.repeat(range(params.test_n_way), params.n_shot))).cuda()
            mini_x_b_i = mini_x_var[:, n_support:,:,:,:].contiguous().view(params.test_n_way * 15, *mini_x.size()[2:])
            mini_x_a_i = mini_x_var[:,:n_support,:,:,:].contiguous().view(params.test_n_way * 5, *mini_x.size()[2:])
            mini_idxs = [row[0].item() for row in mini_y]
            mini_batch_classnames = [return_novel_label_names()[idx] for idx in mini_idxs]


        #print(f'Extracting and saving miniImagenet features for episode {episode_idx}...')
        #extract_and_save_features(model, mini_x_a_i, mini_y_a_i, episode_idx, "miniImagenet", params, "image")
        
  
        mini_y_query = np.repeat(range(params.test_n_way), 15)
        mini_query_labels = torch.tensor(mini_y_query).cuda()
        extract_and_save_features(model, mini_x_b_i, mini_query_labels, episode_idx, "miniImageNet", params, "image")
        

        #mini_text_labels = torch.arange(len(mini_batch_classnames))
    
        #mini_clip_model = load_clip_to_cpu(params)
        #mini_model = CustomCLIP(params, mini_batch_classnames, mini_clip_model, params)
        #mini_model = mini_model.cuda()
        #extract_and_save_features(mini_model, None, mini_text_labels, episode_idx, "miniImagenet", params, "text")
        
    except StopIteration:
        print("miniImagenet loader exhausted, resetting...")
        return

def log(out, log_str):
    out.write(log_str + '\n')
    out.flush()
    print(log_str)


def finetune(args, dataset):
    current_time = datetime.now()

    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    out_file = 'output/log/'+dataset + '/coop_lora/'
    if os.path.exists(out_file):
        out_file = out_file + time_str+'.txt'
        out = open(out_file, 'a')
    else:
        os.makedirs('output/log/'+dataset + '/coop_lora')
        out_file = out_file + time_str+'.txt'
        out = open(out_file, 'a')

    
    n_way = args.test_n_way
    n_support = args.n_shot
    n_query = 15


    total_num = 0
    iter_num = 800 if args.n_shot == 1 else 400
    acc_all = []

    image_size = 224
    split = args.split
    loadfile = os.path.join(args.data_path, dataset, split + '.json')
    datamgr = SetDataManager(image_size, n_way, n_support, n_query, n_eposide = iter_num)
    data_loader = datamgr.get_data_loader(loadfile, aug = args.aug)

    #mini_loader, mini_label_names = initialize_miniimagenet_loader(args)
    


    if dataset == "CropDiseases":
        
        label_names=["Apple___Apple_scab",
                    "Apple___Black_rot",
                    "Apple___Cedar_apple_rust",
                    "Apple___healthy",
                    "Blueberry___healthy",
                    "Cherry_(including_sour)___Powdery_mildew",
                    "Cherry_(including_sour)___healthy",
                    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                    "Corn_(maize)___Common_rust_",
                    "Corn_(maize)___Northern_Leaf_Blight",
                    "Corn_(maize)___healthy",
                    "Grape___Black_rot",
                    "Grape___Esca_(Black_Measles)",
                    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                    "Grape___healthy",
                    "Orange___Haunglongbing_(Citrus_greening)",
                    "Peach___Bacterial_spot",
                    "Peach___healthy",
                    "Pepper,_bell___Bacterial_spot",
                    "Pepper,_bell___healthy",
                    "Potato___Early_blight",
                    "Potato___Late_blight",
                    "Potato___healthy",
                    "Raspberry___healthy",
                    "Soybean___healthy",
                    "Squash___Powdery_mildew",
                    "Strawberry___Leaf_scorch",
                    "Strawberry___healthy",
                    "Tomato___Bacterial_spot",
                    "Tomato___Early_blight",
                    "Tomato___Late_blight",
                    "Tomato___Leaf_Mold",
                    "Tomato___Septoria_leaf_spot",
                    "Tomato___Spider_mites Two-spotted_spider_mite",
                    "Tomato___Target_Spot",
                    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                    "Tomato___Tomato_mosaic_virus",
                    "Tomato___healthy"]

    elif dataset == "EuroSAT":
        label_names=["Annual Crop Land",
                    "Forest",
                    "Herbaceous Vegetation Land",
                    "Highway or Road",
                    "Industrial Buildings",
                    "Pasture Land",
                    "Permanent Crop Land",
                    "Residential Buildings",
                    "River",
                    "Sea or Lake",]
    elif dataset == "ISIC":
        label_names=["Melanoma",
                     "Melanocytic Nevus",
                     "Basal Cell Carcinoma",
                     "Actinic Keratosis",
                     "Benign Keratosis",
                     "Dermatofibroma",
                     "Vascular Lesion"]
    elif dataset == "ChestX":
        label_names=["Atelectasis",
                     "Cardiomegaly",
                     "Effusion",
                     "Infiltration",
                     "Mass",
                     "Nodule",
                     "Pneumothorax"]
    elif dataset == "miniImagenet":
        label_names=return_novel_label_names()


    for i, (x, y) in enumerate(data_loader):
        
        ### no aug: x: [cps, spc + qpc, 3, 224, 224]; y: [cps, spc + qpc]
        ### aug: x: list of [cps, spc + qpc, 3, 224, 224], list len = 5; y: list of [cps, spc + qpc], list len = 5
        ### aug list: first element refers to no aug, each row in y is the same
        
        if params.aug:
            idxs = [row[0].item() for row in y[0]]
        else:
            idxs = [row[0].item() for row in y]
        batch_classnames = [label_names[idx] for idx in idxs]
        
        clip_model = load_clip_to_cpu(args)
        list_lora_layers = apply_lora(args, clip_model)
        coop_lora_model = CustomCLIP(args, batch_classnames, clip_model, params)
        
        #### load source-domain-trained model ####
        if params.ckp_path is not None:
            pre_model_state_dict = torch.load(params.ckp_path)
            model_state_dict = coop_lora_model.state_dict()
            number = 0
            for name in pre_model_state_dict['state'].keys():
                if name in model_state_dict.keys() and "prompt_learner.token_" not in name:# and "lora" not in name:
                    number += 1
                    model_state_dict[name] = pre_model_state_dict['state'][name]
            coop_lora_model.load_state_dict(model_state_dict)
        ##########################################
        
        for name, param in coop_lora_model.named_parameters():
            # param = param.to(torch.float32)
            param = param.type(torch.float16)

        for name, param in coop_lora_model.named_parameters():
            
            if "prompt_learner" not in name and "lora" not in name and "txt_w" not in name and "vpt" not in name and 'txt2visual_proj' not in name and 'ln_1_pos' not in name:
                #print('nograd:',name)
                param.requires_grad_(False)
            #else:
                #print('grad:',name)
        
        total_epoch = args.total_epoch #100
        batch_size = 9999 # use all support samples in each step

        lora_params = get_lora_parameters(coop_lora_model)
        coop_params=[]
        other_params=[]
        for name, param in coop_lora_model.named_parameters():
            if "prompt_learner" in name:
                coop_params.append(param)
            elif "lora" not in name:
                other_params.append(param)
                

        lora_optimizer = torch.optim.AdamW(#[
                                    #   {'params':visual_params, 'lr':2e-4},
                                    #    {'params':text_params, 'lr':0.2}
                                    #],
                                    lora_params, lr=args.lora_lr,
                                    # [{'params':lora_params, 'lr':2e-4},
                                    # {'params':coop_params,'lr':0.002}],                                    
                                    #weight_decay=1e-2, 
                                    #betas=(0.9, 0.999), 
                                    )
        coop_optimizer = torch.optim.SGD(
            coop_params,lr=args.coop_lr,momentum=0.9,weight_decay=5e-4
        )
        other_optimizer = torch.optim.SGD(
            other_params,lr=args.base_lr,momentum=0.9,weight_decay=5e-4
        )
    
        coop_lora_model = coop_lora_model.cuda()

        # ##################
        if params.aug:
            n_query = x[0].size(1) - n_support
            multi_x = x
            sup_x = []; qry_x = []; label = []
            for x in multi_x:
                x = x.cuda()
                x_var = Variable(x)

                support_size = n_way * n_support 
               
                # [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]
                y_a_i = Variable(torch.from_numpy(np.repeat(range(n_way), n_support))).cuda() 
                x_b_i = x_var[:, n_support:,:,:,:].contiguous().view(n_way* n_query, *x.size()[2:]) 
                x_a_i = x_var[:,:n_support,:,:,:].contiguous().view(n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)

                sup_x.append(x_a_i); qry_x.append(x_b_i); label.append(y_a_i)

            x_a_i = torch.cat(sup_x, dim=0) # [num_aug * cps * spc, c, h, w]
            y_a_i = torch.cat(label, dim=0) # [num_aug * cps * spc]
            #x_a_i = sup_x[1] # [cps * spc, c, h, w]
            #y_a_i = label[1] # [cps * spc]

            x_b_i = qry_x[0] # [cps * qpc, c, h, w], only use the no-aug version
           
        else:
            n_query = x.size(1) - n_support
            x = x.cuda()
            x_var = Variable(x)

            support_size = n_way * n_support 
           
            # [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]
            y_a_i = Variable(torch.from_numpy(np.repeat(range(n_way), n_support))).cuda() 
            x_b_i = x_var[:, n_support:,:,:,:].contiguous().view(n_way* n_query, *x.size()[2:]) 
            x_a_i = x_var[:,:n_support,:,:,:].contiguous().view(n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)

        support_size = x_a_i.shape[0]
        
        epoch_loss = []
        coop_lora_model.train() 
  


        for epoch in range(total_epoch):
            
            params.aux_container['support_epoch']= epoch
            
            rand_id = np.random.permutation(support_size)
            
            
            for j in range(0, support_size, batch_size):
          
                coop_optimizer.zero_grad()#+
                lora_optimizer.zero_grad()#+
                other_optimizer.zero_grad()#+
                
                selected_id = torch.from_numpy(rand_id[j: min(j+batch_size, support_size)]).cuda()
                
                z_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id]  # support label
                params.aux_container['label'] = y_batch
        

                logits = coop_lora_model(z_batch, params=params)#, sp_img=x_a_i)
                loss = F.cross_entropy(logits, y_batch)#+ params.loss_rate*F.cross_entropy(pre_logits, y_batch)
       

                loss.backward()
              
                coop_optimizer.step()
                lora_optimizer.step()
                other_optimizer.step()

        coop_lora_model.eval()

        

        y_query = np.repeat(range(n_way), n_query)
        params.aux_container['label'] = torch.tensor(y_query).cuda()


        
        output = coop_lora_model(x_b_i.cuda(), params=params)#, sp_img=x_a_i.cuda())

      
      



        topk_scores, topk_labels = output.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        
        log_str = dataset + ' iteration %d/%d :'%(i, iter_num)

        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        acc_all.append((correct_this/ count_this *100))
        
        acc_all_np = np.asarray(acc_all); acc_mean = np.mean(acc_all_np); acc_std = np.std(acc_all_np)
        log_str = log_str + ' Query Acc = %4.2f, Avg Acc = %4.2f%% +- %4.2f%%'%(correct_this/ count_this *100, acc_mean, 1.96 * acc_std/np.sqrt(total_num+1))
        log(out, log_str)



        if total_num % 100 == 0:
            acc_all_np = np.asarray(acc_all)
            acc_mean = np.mean(acc_all_np)
            acc_std = np.std(acc_all_np)
            #print(dataset, '|', total_num, '/', iter_num, ' Acc = %4.2f%% +- %4.2f%%' %(acc_mean, 1.96 * acc_std/np.sqrt(total_num+1)))
            print_str = str(dataset) + ' | ' + str(total_num) +  ' / ' + str(iter_num) + ' Acc = %4.2f%% +- %4.2f%%'%(acc_mean, 1.96 * acc_std/np.sqrt(total_num+1)); log(out, print_str)

        total_num = total_num + 1
        ####################
        
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    log_str = dataset[:4] + ' %d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num))
    log(out, log_str)


# --- main ---
if __name__ == '__main__':

    # parse argument
    params = parse_args('test')
    
    seed = params.seed
    print("set seed = %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print('FineTune! {} shots on {} dataset'.format(params.n_shot, params.dataset))
    remove_featurefile = True

    if(params.dataset == 'multi'):
        all_dataset = ['CropDiseases', 'EuroSAT', 'ISIC2018', 'ChestX']#, 'cub', 'cars', 'places', 'plantae']
        # all_dataset = ['cub', 'cars', 'places', 'plantae']
        for test_dataset in all_dataset:
            params.dataset = test_dataset
            acc = finetune(params, test_dataset)

    else:
        acc = finetune(params, params.dataset)
