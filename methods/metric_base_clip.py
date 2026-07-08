# This code is modified from https://github.com/jakesnell/prototypical-networks

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MetaCLIP(nn.Module):
    def __init__(self, clip_model, n_way, n_support):
        super(MetaCLIP, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.clip_model = clip_model
        
    def set_forward(self, x, t_all, select_class, params=None):
        z_support, z_query = self.parse_feature(x)
        z_support = z_support.contiguous()
        z_query = z_query.contiguous()
        z_support = z_support.view(self.n_way*self.n_support, -1)
        z_query = z_query.view(self.n_way*self.n_query, -1)
        
        t_all = t_all.cuda()
        z_query = z_query / z_query.norm(dim=-1, keepdim=True)
        t_all = t_all / t_all.norm(dim=-1, keepdim=True)
        
        logit_scale = self.clip_model.logit_scale.exp()

        logits = logit_scale * z_query @ t_all.t()
        '''
        cross_w = params.aux_container['cross_w']
        cross_w = torch.cat([cross_w, torch.tensor([1.]).cuda()])
        w = cross_w.abs() / cross_w.abs().sum() # [num_block]
        all_logits = logit_scale * z_query.unsqueeze(0) @ t_all.permute(1, 2, 0) # [num_block, b, num_class]
        logits = (all_logits * w.unsqueeze(-1).unsqueeze(-1)).sum(0)
        '''
        
        return logits
    
    def forward(self, x):
        pass
    
    def parse_feature(self, x):
        x = x.cuda()
        z_all = x
        
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        
        return z_support, z_query
        
    

# class MetaCLIP(MetaTemplate):
#     def __init__(self, model_func,  n_way, n_support, tf_path=None):
#         super(MetaCLIP, self).__init__(model_func,  n_way, n_support, tf_path=tf_path)
#         self.loss_fn = nn.CrossEntropyLoss()

#     def reset_modules(self):
#         return

#     def set_forward(self,x,is_feature=False):
#         z_support, z_query = self.parse_feature(x,is_feature)
#         z_support = z_support.contiguous()
#         z_query = z_query.contiguous()
#         """
#         #### cos similarity is better ####
#         z_support = F.normalize(z_support, p=2, dim=-1)
#         z_query = F.normalize(z_query, p=2, dim=-1)
        
#         #### simplest transductive calibration ####
#         # z_support: [cps, spc, d], z_query: [cps, qpc, d]
#         r_query = z_query.view(self.n_way * self.n_query, -1) # [cps*qpc, d]
#         r_support = z_support.view(self.n_way * self.n_support, -1) # [cps*spc, d]
#         #r_all_data = torch.cat([r_support, r_query], dim=0) # [cps*spc+cps*qpc, d]
#         r_all_data = r_support
#         #r_all_data = r_query
#         r_mean = r_all_data.mean(dim=0) # [d]
#         r_std = r_all_data.std(dim=0) # [d]
        
#         r_mean = r_mean.unsqueeze(0).unsqueeze(0); r_std = r_std.unsqueeze(0).unsqueeze(0) # [1, 1, d]
        
#         z_support = z_support - r_mean
#         z_query = z_query - r_mean
        
#         #z_support = (z_support - r_mean) / (r_std + 1e-5)
#         #z_query = (z_query - r_mean) / (r_std + 1e-5)

#         z_support = F.normalize(z_support, p=2, dim=-1)
#         z_query = F.normalize(z_query, p=2, dim=-1)
        
#         ###########################################
#         """
#         z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1) #the shape of z is [n_data, n_dim]
#         z_query = z_query.view(self.n_way * self.n_query, -1)

#         dists = euclidean_dist(z_query, z_proto)

#         scores = -dists
#         return scores

#     def get_distance(self,x,is_feature = False):
#         z_support, z_query = self.parse_feature(x,is_feature)
#         z_support = z_support.contiguous()
#         z_proto = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
#         z_query = z_query.contiguous().view(self.n_way* self.n_query, -1)
#         return euclidean_dist(z_proto, z_proto)[0, :5].cpu().numpy()

#     def set_forward_loss(self, x):
#         y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
#         y_query = y_query.cuda()
#         scores = self.set_forward(x)
#         loss = self.loss_fn(scores, y_query)
#         return scores, loss


# def euclidean_dist( x, y):
#     # x: N x D
#     # y: M x D
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     assert d == y.size(1)

#     x = x.unsqueeze(1).expand(n, m, d)
#     y = y.unsqueeze(0).expand(n, m, d)

#     return torch.pow(x - y, 2).sum(2)
#     #return torch.pow(F.normalize(x, p=2, dim=-1) - F.normalize(y, p=2, dim=-1), 2).sum(2)
