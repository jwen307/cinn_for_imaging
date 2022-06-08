"""
Train CINNReconstructor on 'fastMRI'.
"""

import os
import time
import torchvision

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

#from cinn_for_imaging.reconstructors.mri_reconstructor import CINNReconstructor
#from cinn_for_imaging.datasets.fast_mri.data_util import FastMRIDataModule
import sys
sys.path.append("..") 
from reconstructors.mri_reconstructor import CINNReconstructor
from datasets.fast_mri.data_util import FastMRIDataModule

#Show multiple images in a grid from a tensor
def show_tensor_imgs(plot_imgs,nrow = 5, return_grid = False, **kwargs):
    ''' 
    Show tensor images (with values between -1 and 1) in a grid
    
    plot_imgs: (batch_size, num_channels, height, width) [Tensor] tensor of imgs with values between -1 and 1
    nrows: Number of imgs to include in a row
    '''
    
    #Put the images in a grid and show them
    if (plot_imgs[0].dtype == torch.int32) or (plot_imgs[0].dtype == torch.uint8):
        grid = torchvision.utils.make_grid(plot_imgs, nrow = int(nrow), scale_each=False, normalize=False)
        
    else:
        grid = torchvision.utils.make_grid(plot_imgs, nrow = int(nrow), scale_each=True, normalize=True)
        
    if not return_grid:
        f = plt.figure()
        f.set_figheight(15)
        f.set_figwidth(15)
        plt.imshow(grid.permute(1, 2, 0).numpy())
    
        #Use a custom title
        if 'title' in kwargs:
            plt.title(kwargs['title'])

    else:
        return grid

#%% setup the dataset
dataset = FastMRIDataModule(num_data_loader_workers=os.cpu_count(),
                            batch_size=8)
dataset.prepare_data()
dataset.setup()

#%% path for logging and saving the model
experiment_name = 'cinn'
path_parts = ['..', 'experiments', 'fast_mri', experiment_name]
log_dir = os.path.join(*path_parts)

#%% configure the Pytorch Lightning trainer. 
# Visit https://pytorch-lightning.readthedocs.io/en/stable/trainer.html
# for all available trainer options.

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
)


lr_monitor = LearningRateMonitor(logging_interval=None) 


tb_logger = pl_loggers.TensorBoardLogger(log_dir)

trainer_args = {'gpus': 1,
                'default_root_dir': log_dir,
                'callbacks': [lr_monitor, checkpoint_callback],
                'benchmark': True,
                'fast_dev_run': False,
                'gradient_clip_val': 1.0,
                'logger': tb_logger,
                'precision': 32,
                'limit_train_batches': 0.25,
                'limit_val_batches': 0.25,
                'terminate_on_nan': True}#,
                #'limit_train_batches': 0.1,
                #'limit_val_batches': 0.1}#,
                #'overfit_batches':10}

#%% create the reconstructor
reconstructor = CINNReconstructor(
    in_ch=1, 
    img_size=(320, 320),
    max_samples_per_run=100,
    trainer_args=trainer_args,
    log_dir=log_dir)

#%% change some of the hyperparameters of the reconstructor
reconstructor.batch_size = 12
reconstructor.epochs = 500

reconstructor.sample_distribution = 'normal'
reconstructor.conditioning = 'ResNetCond'
reconstructor.coupling = 'affine'
reconstructor.cond_fc_size = 64
reconstructor.num_blocks = 5
reconstructor.num_fc = 3  # TODO does nothing, replace by reconstructor.num_fc_blocks (= 2)?
reconstructor.permutation = '1x1'
reconstructor.cond_conv_channels = [4, 16, 16, 16, 32, 32, 32]
reconstructor.train_noise = (0., 0.005)
reconstructor.use_act_norm = True

#%% train the reconstructor. Checkpointing and logging is enabled by default.
checkpoint = '../experiments/fast_mri/cinn/default/version_4/checkpoints/epoch=200.ckpt'
start = time.time()
reconstructor.train(dataset, checkpoint_path = checkpoint)
end = time.time()
print('Total Training Time: {0}'.format(end-start))


#%% Get some samples
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

reconstructor.model.eval()

imgs = []
for i, data in tqdm(enumerate(dataset.val_dataloader())):
    
    if i < 10:
        continue
    
    obs, gt, mean, std, fname, slice_num, max_value = data
    
    
    obs = obs.to('cuda')
    gt = gt.to('cuda')

    with torch.no_grad():
        reco = reconstructor._reconstruct(obs)
    
    imgs.append(gt.cpu())
    reco = torch.tensor(np.asarray(reco)).resize(1,1,320,320)
    imgs.append(obs.cpu())
    imgs.append(reco)
    
    
    
    if i == 15:
        break
    
show_tensor_imgs(torch.cat(imgs,dim=0),nrow=3)


#%% Get the metrics for recovery with the median
from benchmark import recon_metrics



metrics = recon_metrics.Metrics(None)

reconstructor.model.to('cuda')
reconstructor.model.eval()
dataset = FastMRIDataModule(num_data_loader_workers=os.cpu_count(),
                            batch_size=1)
dataset.prepare_data()
dataset.setup()

for i, data in tqdm(enumerate(dataset.val_dataloader())):
    obs, gt, mean, std, fname, slice_num, max_value = data
    
    obs = obs.to('cuda')
    gt = gt.to('cuda')
    
    
    batch_size, _, _, _ = obs.size()
    
    with torch.no_grad():
        reco, reco_std = reconstructor._reconstruct(obs,return_std=True)
        
    for j in range(batch_size):
        metrics.push(gt[j].cpu().numpy(), np.asarray(reco).reshape(1,320,320))
        
print(metrics)