import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import random
import os

#from data.datamgr import SetDataManager
from data.datamgr_aug import SetDataManager # use this line if aug
from utils import *
import torch.nn.functional as F

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from options.options_coop import parse_args
from optimizer import build_optimizer

##### text tokenizer #####
_tokenizer = _Tokenizer()

##### build clip model and load pretrained params #####
def load_clip_to_cpu(args):
    backbone_name = args.model_name
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

##### text branch #####
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, params=None):
        params.aux_container['txt_pos'] = self.positional_embedding.type(self.dtype).unsqueeze(0).permute(1, 0, 2)
        params.aux_container['tokenized_prompts_argmax'] = tokenized_prompts.argmax(dim=-1)
       
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x,params=params)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        xs = []
        for i in range(12):
            xs.append(self.ln_final(params.aux_container['text_block_output_%d'%i].permute(1, 0, 2)).type(self.dtype)[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection)
        params.aux_container['txt_block_feats'] = xs
        return x


##### prompt learner #####
class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.n_ctx  # context 
        ctx_init = args.ctx_init # context initialization
        dtype = clip_model.dtype
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


##### CoOp #####
class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model, params):
        super().__init__()
        self.prompt_learner = PromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image,  label=None,params=None, sp_img=None):
       

        prompts = self.prompt_learner()        
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts, params=params)
        params.aux_container['txt_feat_final'] = text_features #8
        params.aux_container['txt_feat'] = params.aux_container['txt_block_feats'][10]#BEST
        
        
        image_features = self.image_encoder(image.type(self.dtype), params=params)
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

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        if sp_img is not None:
            logits = logits + params.aux_param * image_features @ sp_img_feats.t() * logit_scale

        return logits


##### episode finetune and test on query set #####
def finetune(args, dataset): 

    ##### few-shot params #####
    n_way = args.test_n_way
    n_support = args.n_shot
    n_query = 15

    iter_num = 100 if args.n_shot == 1 else 400
    total_num = 0
    acc_all = []   

    ##### data loader #####
    ##### with a episode way [n_way * (n_support + n_query)] #####
    image_size = 224
    split = args.split
    loadfile = os.path.join(args.data_path, dataset, split + '.json')
    datamgr= SetDataManager(image_size, n_way, n_support, n_query, n_eposide = iter_num)
    data_loader= datamgr.get_data_loader(loadfile, aug = args.aug)
    
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
        # label_names=["Apple Apple scab",
        #             "Apple Black Rot",
        #             "Apple Cedar Apple Rust",
        #             "Apple Healthy",
        #             "Blueberry Healthy",
        #             "Cherry Powdery Mildew",
        #             "Cherry Healthy",
        #             "Corn Cercospora Leaf Spot Gray Leaf Spot",
        #             "Corn Common Rust",
        #             "Corn Northern Leaf Blight",
        #             "Corn Healthy",
        #             "Grape Black rot",
        #             "Grape Esca",
        #             "Grape Leaf Blight",
        #             "Grape Healthy",
        #             "Orange Haunglongbing",
        #             "Peach Bacterial Spot",
        #             "Peach Healthy",
        #             "Pepper Bell Bacterial Spot",
        #             "Pepper Bell Healthy",
        #             "Potato Early Blight",
        #             "Potato Late Blight",
        #             "Potato Healthy",
        #             "Raspberry Healthy",
        #             "Soybean Healthy",
        #             "Squash Powdery Mildew",
        #             "Strawberry Leaf Scorch",
        #             "Strawberry Healthy",
        #             "Tomato Bacterial_spot",
        #             "Tomato Early Blight",
        #             "Tomato Late Blight",
        #             "Tomato Leaf Mold",
        #             "Tomato Septoria Leaf Spot",
        #             "Tomato Spider-mites Two-spotted Spider Mite",
        #             "Tomato Target Spot",
        #             "Tomato Tomato Yellow Leaf Curl Virus",
        #             "Tomato Tomato Mosaic Virus",
        #             "Tomato Healthy"]
        # args.ctx_init = "a crop disease photo of a"
    elif dataset == "EuroSAT":
        # label_names=["AnnualCrop",
        #              "Forest",
        #              "HerbaceousVegetation",
        #              "Highway",
        #              "Industrial",
        #              "Pasture",
        #              "PermanentCrop",
        #              "Residential",
        #              "River",
        #              "SeaLake"]
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
        # args.ctx_init = "a centered satellite photo of a"
    elif dataset == "ISIC":
        # label_names=["MEL",
        #              "NV",
        #              "BCC",
        #              "AKIEC",
        #              "BKL",
        #              "DF",
        #              "VASC"]
        label_names=["Melanoma",
                     "Melanocytic Nevus",
                     "Basal Cell Carcinoma",
                     "Actinic Keratosis",
                     "Benign Keratosis",
                     "Dermatofibroma",
                     "Vascular Lesion"]
        # args.ctx_init = "a skin lession photo of a"
    else:
        label_names=["Atelectasis",
                     "Cardiomegaly",
                     "Effusion",
                     "Infiltration",
                     "Mass",
                     "Nodule",
                     "Pneumothorax"]
        # args.ctx_init = "a chest x-ray photo of a"
        
    for _, (x, y) in enumerate(data_loader):
        ##### current episode classnames #####
        if params.aug:
            idxs = [row[0].item() for row in y[0]]
        else:
            idxs = [row[0].item() for row in y]
        curr_classnames = [label_names[idx] for idx in idxs]
        # print(curr_class_names)
        
        ##### use pretrained clip to build coop model #####
        clip_model = load_clip_to_cpu(args)
        coop_model = CustomCLIP(args, curr_classnames, clip_model, params)
        for name, param in coop_model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        coop_model.cuda()

        ##### few shot data #####
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
    
             # use all support samples in each step
            support_size = n_way * n_support 


        
            y_a_i = Variable(torch.from_numpy(np.repeat(range(n_way), n_support))).cuda() # (25,) # supp_label
            x_b_i = x_var[:, n_support:,:,:,:].contiguous().view(n_way* n_query, *x.size()[2:]) # query
            x_a_i = x_var[:,:n_support,:,:,:].contiguous().view(n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224) # support
            
        ##### optimizer #####
        coop_optim = build_optimizer(coop_model.prompt_learner, args)
        
        ##### model training use support set #####
        total_epoch = 100
        batch_size = 9999
        coop_model.train()
        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size)
            
            for j in range(0, support_size, batch_size):
                coop_optim.zero_grad()
                
                selected_id = torch.from_numpy(rand_id[j: min(j+batch_size, support_size)]).cuda()
                
                z_batch = x_a_i[selected_id]  # support feature
                  
                # #### simple augmentation ######
                # import torchvision
                # flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
                # z_batch = flip(z_batch)
                #####################################
                
                y_batch = y_a_i[selected_id]  # support label
             
                params.aux_container['label'] = y_batch
                output = coop_model(z_batch, params=params)
                loss = F.cross_entropy(output, y_batch)
                loss.backward()
                coop_optim.step()

        coop_model.eval()
        
        ##### model evalution use query set #####
        y_query = np.repeat(range(n_way), n_query )
        params.aux_container['label'] = torch.tensor(y_query).cuda()
        output = coop_model(x_b_i.cuda(), params=params)
        
        topk_scores, topk_labels = output.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        acc_all.append((correct_this/ count_this *100))

        if total_num % 10 == 0:
            #print (dataset, '|', total_num, '/', iter_num, 'Acc = %4.2f'%(correct_this / count_this * 100))
            acc_all_np = np.asarray(acc_all)
            acc_mean = np.mean(acc_all_np)
            acc_std = np.std(acc_all_np)
            print(dataset, '|', total_num, '/', iter_num, ' Acc = %4.2f%% +- %4.2f%%' %(acc_mean, 1.96 * acc_std/np.sqrt(total_num+1)))

        total_num = total_num + 1
        
        ####################

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print(dataset, '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

# --- main ---
if __name__ == '__main__':
  # random seed
  seed = 1
  print("set seed = %d" % seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  # parse argument
  params = parse_args('test')
  print('FineTune! {} shots on {} dataset'.format(params.n_shot, params.dataset))

  if(params.dataset == 'multi'):
    all_dataset = ['CropDiseases', 'EuroSAT', 'ISIC2018', 'ChestX']
    # all_dataset = ['cub', 'cars', 'places', 'plantae']
    for test_dataset in all_dataset:
        acc = finetune(params, test_dataset)

  else:
    acc = finetune(params, params.dataset)
