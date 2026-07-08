import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import random
import os

import torch.optim

from data.datamgr import SetDataManager # use this line if aug is not used
#from data.datamgr_aug import SetDataManager # use this line if aug
from options.options_coop_lora import parse_args

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

class TextEncoder(nn.Module):  # 文本分支
    def __init__(self, clip_model, params):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

        #self.alphas = torch.nn.Parameter(torch.zeros(12 - int(params.aux_param)) - 4.6) # 0.01 after softplus
        #self.alphas = torch.zeros(12 - int(params.aux_param)) + params.aux_param2


    def forward(self, prompts, tokenized_prompts, params=None):
        # print(prompts[0][0][0])
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, params=params)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1) - 1] @ self.text_projection
        '''
        xs = []
        for i in range(12):
            xs.append(self.ln_final(params.aux_container['text_block_output_%d'%i].permute(1, 0, 2)).type(self.dtype)[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection)
        params.aux_container['txt_block_feats'] = xs
        '''
        return x
        '''
        xs = []
        for i in range(int(params.aux_param), 12):
            xs.append(self.ln_final(params.aux_container['text_block_output_%d'%i].permute(1, 0, 2)).type(self.dtype)[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection)
        xs = torch.stack(xs) # [num_block, num_class, c]
        xs = xs.permute(1, 0, 2) # [num_class, num_block, c]
        #return xs
        xs = F.normalize(xs, dim=-1).half()
        x = xs.mean(dim=1)
        #alphas = F.softplus(self.alphas).half().cuda()
        #ws = alphas / alphas.sum()
        #x = (ws.unsqueeze(0).unsqueeze(-1).cuda() * xs).sum(0).half() # [num_class, c]
        return x
        '''
        '''
        xs = []
        for i in range(int(params.aux_param), 12):
            xs.append(self.ln_final(params.aux_container['text_block_output_%d'%i].permute(1, 0, 2)).type(self.dtype)[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection)
        xs = torch.stack(xs) # [num_block, num_class, c]
        xs = F.normalize(xs, dim=-1).half()
        if self.training:
            epsilon = 0.000001
            #self.distr = torch.distributions.dirichlet.Dirichlet(F.softplus(self.alphas.cuda()) + epsilon)
            self.distr = torch.distributions.dirichlet.Dirichlet(self.alphas.cuda())
            ws = self.distr.rsample().half().cuda()
            xs = xs + torch.randn_like(xs).cuda() * F.softplus(self.alphas.cuda()).unsqueeze(-1).unsqueeze(-1) * params.aux_param
            xs = F.normalize(xs, dim=-1).half()           
        else:
            #alphas = F.softplus(self.alphas).half().cuda()
            alphas = self.alphas
            ws = alphas / alphas.sum()
        x = (ws.unsqueeze(-1).unsqueeze(-1).cuda() * xs).sum(0).half() # [num_class, c]
        return x
        '''




class PromptLearner(nn.Module):  # prompt learner
    def __init__(self, args, classnames, clip_model, ctx):
        super().__init__()
        n_cls = len(classnames) 
        n_ctx = args.n_ctx  # context 
        ctx_init = ctx #args.ctx_init # context initialization
        # dtype = clip_model.dtype
        dtype = torch.float16
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = args.img_size
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init: # 用给定的words初始化
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:  # 随机初始化
            # random initialization
            if args.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of context words (tokens): {n_ctx}")

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
        ctx_a = 'a photo of wet cloth, not dark cloth'
        ctx_b = 'a photo of dark table'
        self.ctx_a = ctx_a; self.ctx_b = ctx_b
        self.prompt_learner_a = PromptLearner(args, classnames, clip_model, ctx_a)
        self.prompt_learner_b = PromptLearner(args, classnames, clip_model, ctx_b)
        self.tokenized_prompts_a = self.prompt_learner_a.tokenized_prompts
        self.tokenized_prompts_b = self.prompt_learner_b.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, params)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        #ids = [0, 1]#, 10, 9, 8, 7]
        #self.cross_w = nn.Parameter(F.one_hot(torch.tensor(11), 12).to(self.dtype) + torch.randn(12).to(self.dtype) * 0.00001) # [num_block]
        #self.cross_w = nn.Parameter(torch.zeros(11).to(self.dtype)) # [num_block]
        #self.cross_w = nn.Parameter(F.one_hot(torch.tensor(ids), 12).to(self.dtype).sum(0)) # [num_block]


    def forward(self, image, label=None, params=None, sp_img=None):
        # image_features: [b, c]; text_features: [num_class, c]; label: [b]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            
            prompts_a = self.prompt_learner_a()
            tokenized_prompts_a = self.tokenized_prompts_a
            prompts_b = self.prompt_learner_b()
            tokenized_prompts_b = self.tokenized_prompts_b

            text_features_a = self.text_encoder(prompts_a, tokenized_prompts_a, params)
            text_features_b = self.text_encoder(prompts_b, tokenized_prompts_b, params)
            print(float((F.normalize(text_features_a, dim=-1) @ F.normalize(text_features_b, dim=-1).t()).mean()), '  :  "', self.ctx_a, '"  vs. " ', self.ctx_b, '"'); exit()


            # image_features = self.image_encoder(image.type(self.dtype))
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
        
        """
        if self.training:
            
            text_y_features = F.one_hot(label, text_features.shape[0]).cuda().half() @ text_features
            diff = (image_features - text_y_features).abs().detach() # [b, c]
            '''
            diff -= diff.min(1, keepdim=True)[0]
            diff /= diff.max(1, keepdim=True)[0]
            print(diff, torch.histogram(diff.float().cpu(), 10)); exit()
            image_features = image_features + torch.randn_like(image_features).cuda() * diff * 0.01
            '''
            topk_val, topk_idx = torch.topk(diff, int(params.aux_param * diff.shape[-1])) # [b, topk], [b, topk]
            topk_mask = F.one_hot(topk_idx, diff.shape[-1]).sum(dim=1).cuda() # [b, topk, c] -> [b, c]
            image_features = image_features + torch.randn_like(image_features).cuda() * topk_mask * 0.1
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            #text_features = text_features + torch.randn_like(text_features).cuda() * 0.1
            #text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        """


        #img_tokens = image_features[:, 1:, :]; cls_tokens = image_features[:, 0, :]
        #logit_scale = self.logit_scale.exp()
        logit_scale = 4 if params.n_shot == 5 else 2

        logits = logit_scale * image_features @ text_features.t()
        '''
        cross_w = self.cross_w
        cross_w = torch.cat([self.cross_w, torch.tensor([1.]).cuda()])
        w = cross_w.abs() / cross_w.abs().sum() # [num_block]
        all_logits = logit_scale * image_features.unsqueeze(0) @ text_features.permute(1, 2, 0) # [num_block, b, num_class]
        logits = (all_logits * w.unsqueeze(-1).unsqueeze(-1)).sum(0)
        '''

        if sp_img is not None:
            logits = logits + params.aux_param * image_features @ sp_img_feats.t() * logit_scale
        '''
        if self.training:
            txt_tokens = params.aux_container['visual_feat_map'][:, -text_features.shape[0]:, :] # [b, cps, c]
            print(torch.norm(txt_tokens, dim=-1)); exit()
            txt_tokens = F.normalize(txt_tokens, dim=-1)
            token_logits = image_features.unsqueeze(1) @ txt_tokens.permute(0, 2, 1) # [b, 1, cps]
            token_logits = token_logits.squeeze() * logit_scale
            params.aux_container['token_logits'] = token_logits       
            return token_logits
        '''
        return logits



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
        '''
        label_names=["Apple (Apple scab)",
                    "Apple (Black rot)",
                    "Apple (Cedar apple rust)",
                    "Apple (healthy)",
                    "Blueberry (healthy)",
                    "Cherry (including sour) (Powdery mildew)",
                    "Cherry (including sour) (healthy)",
                    "Corn (maize) (Cercospora leaf spot Gray leaf spot)",
                    "Corn (maize) (Common rust)",
                    "Corn (maize) (Northern Leaf Blight)",
                    "Corn (maize) (healthy)",
                    "Grape (Black rot)",
                    "Grape (Esca, Black_Measles)",
                    "Grape (Leaf blight, Isariopsis Leaf Spot)",
                    "Grape (healthy)",
                    "Orange (Haunglongbing, Citrus greening)",
                    "Peach (Bacterial spot)",
                    "Peach (healthy)",
                    "Pepper, bell (Bacterial spot)",
                    "Pepper, bell (healthy)",
                    "Potato (Early blight)",
                    "Potato (Late blight)",
                    "Potato (healthy)",
                    "Raspberry (healthy)",
                    "Soybean (healthy)",
                    "Squash (Powdery mildew)",
                    "Strawberry (Leaf scorch)",
                    "Strawberry (healthy)",
                    "Tomato (Bacterial spot)",
                    "Tomato (Early blight)",
                    "Tomato (Late blight)",
                    "Tomato (Leaf Mold)",
                    "Tomato (Septoria leaf spot)",
                    "Tomato (Spider mites Two-spotted spider mite",
                    "Tomato (Target Spot)",
                    "Tomato (Tomato Yellow Leaf Curl Virus)",
                    "Tomato (Tomato mosaic virus)",
                    "Tomato (healthy)"]
    '''
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

    # classname_file = args.data_path + "/" + dataset + "/classname.txt"
    # label_names = []
    # for name in open(classname_file):
    #     label_names.append(name)

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
            if "prompt_learner" not in name and "lora" not in name and "txt_w" not in name and "vpt" not in name and 'txt2visual_proj' not in name:
                param.requires_grad_(False)
                
        total_epoch = args.total_epoch
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
            rand_id = np.random.permutation(support_size)
            
            for j in range(0, support_size, batch_size):
                coop_optimizer.zero_grad()
                lora_optimizer.zero_grad()
                other_optimizer.zero_grad()
                
                selected_id = torch.from_numpy(rand_id[j: min(j+batch_size, support_size)]).cuda()
                
                z_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id]  # support label
                params.aux_container['label'] = y_batch

                # logits = forward(args, model, batch_classnames, z_batch)
                logits = coop_lora_model(z_batch, params=params)#, sp_img=x_a_i)
                loss = F.cross_entropy(logits, y_batch)
                # print(loss)

                #token_logits = params.aux_container['token_logits']
                #token_loss = F.cross_entropy(token_logits, y_batch)
                #loss = loss + token_loss * params.aux_param2


                epoch_loss.append(loss.cpu().detach().numpy())
                loss.backward()
                #print(coop_lora_model.text_encoder.alphas.grad); exit()
                coop_optimizer.step()
                lora_optimizer.step()
                other_optimizer.step()

        # if args.save_lora:
        #     save_lora(args, list_lora_layers)
        coop_lora_model.eval()

        # [0,0,0,....,1,1,1,......4,4,4,...]
        y_query = np.repeat(range(n_way), n_query)
        params.aux_container['label'] = torch.tensor(y_query).cuda()

        #print(F.softplus(coop_lora_model.text_encoder.alphas) / F.softplus(coop_lora_model.text_encoder.alphas).sum()); exit()
       
        output = coop_lora_model(x_b_i.cuda(), params=params)#, sp_img=x_a_i.cuda())

        # output = evaluate(args, model, batch_classnames, x_b_i.cuda())
        topk_scores, topk_labels = output.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        
        log_str = dataset + ' iteration %d/%d :'%(i, iter_num)

        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        acc_all.append((correct_this/ count_this *100))
        
        acc_all_np = np.asarray(acc_all); acc_mean = np.mean(acc_all_np); acc_std = np.std(acc_all_np)
        log_str = log_str + ' Query Acc = %4.2f, Avg Acc = %4.2f%% +- %4.2f%%'%(correct_this/ count_this *100, acc_mean, 1.96 * acc_std/np.sqrt(total_num+1))
        log(out, log_str)
        '''
        for ii in range(12):
            print(np.mean(params.aux_container['values_%d'%ii]))
        print('*'*40)
        '''
        
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
