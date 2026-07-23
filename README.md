
# Official implementation of paper: Improving CLIP Adaptation by Breaking Tail Alignment for Source-Free Cross-Domain Few-Shot Learning (ICML 2026)

# 1. About this code

This code is for the paper: Improving CLIP Adaptation by Breaking Tail Alignment for Source-Free Cross-Domain Few-Shot Learning (ICML 2026)

# 2. Setup and datasets

## 2.1. Setup

An Anaconda environment is recommended:

```
conda create --name py36 python=3.6
conda activate py36
conda install pytorch torchvision -c pytorch
pip3 install scipy>=1.3.2
pip3 install tensorboardX>=1.4
pip3 install h5py>=2.9.0
pip3 install clip
```


## 2.2. Datasets

Five datasets, including miniImagenet, CropDiseases, EuroSAT, ISIC2018, and ChestX, are used.

Following the [FWT-repo](https://github.com/hytseng0509/CrossDomainFewShot) and [cdfsl-benchmark-repo
](https://github.com/IBM/cdfsl-benchmark)to download and set up all datasets.

Remember to modify your dataset dir in the 'options/options_coop_lora.py'.

# 3. Usage

## 3.1. Fine-tuning

```
#e.g., ISIC
python3 coop_lora_trainer.py -r 16 -alpha 8 -lora_lr 2e-4 -coop_lr 2e-3 -base_lr 0.001 -dataset ISIC  -n_shot 5 -add_ratio 0.3 -minus_ratio 0.2 
```


## 3.2. Data augmentation

To enable the data augmentation code in data/datamgr_aug.py, replace the import in coop_lora_trainer.py (from data.datamgr to data.datamgr_aug) and also pass -aug as a command‑line argument when running the script.


```
#e.g., ISIC
python3 coop_lora_trainer.py -r 16 -alpha 8 -lora_lr 2e-4 -coop_lr 2e-3 -base_lr 0.001 -dataset ISIC  -n_shot 5 -add_ratio 0.3 -minus_ratio 0.2  -aug
```
