import torch
import numpy as np
import h5py

class SimpleHDF5Dataset:
  def __init__(self, file_handle = None):
    if file_handle == None:
      self.f = ''
      self.all_feats_dset = []
      self.all_labels = []
      self.total = 0
    else:
      self.f = file_handle
      self.all_vision_feats_dset = self.f['all_vision_feats'][...]
      self.all_text_feats_dset = self.f['all_text_feats'][...]
      self.all_labels = self.f['all_labels'][...]
      self.total = self.f['count'][0]
  def __getitem__(self, i):
    return torch.Tensor(self.all_vision_feats_dset[i,:]), int(self.all_labels[i])

  def __len__(self):
    return self.total


def init_loader(filename):
  with h5py.File(filename, 'r') as f:
    fileset = SimpleHDF5Dataset(f)

  vision_feats = fileset.all_vision_feats_dset
  text_feats = fileset.all_text_feats_dset
  labels = fileset.all_labels
  while np.sum(vision_feats[-1]) == 0:
    vision_feats  = np.delete(vision_feats,-1,axis = 0)
    labels = np.delete(labels,-1,axis = 0)

  class_list = np.unique(np.array(labels)).tolist()
  inds = range(len(labels))

  cl_data_file = {}
  for cl in class_list:
    cl_data_file[cl] = []
  for ind in inds:
    cl_data_file[labels[ind]].append( vision_feats[ind])

  return cl_data_file, text_feats
