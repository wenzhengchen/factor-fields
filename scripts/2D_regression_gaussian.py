# %%
import torch,imageio,sys,time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.append('..')
from models.FactorFields import FactorFields 

from utils import SimpleSampler
from dataLoader import dataset_dict
from torch.utils.data import DataLoader
imageio.plugins.freeimage.download()

# device = 'cuda'
# torch.cuda.set_device(0)

# load_ext autoreload
# autoreload 2

# %%
def PSNR(a,b):
    if type(a).__module__ == np.__name__:
        mse = np.mean((a-b)**2)
    else:
        mse = torch.mean((a-b)**2).item()
    psnr = -10.0 * np.log(mse) / np.log(10.0)
    return psnr

@torch.no_grad()
def eval_img(reso, chunk=10240):
    y = torch.arange(0, reso[0])
    x = torch.arange(0, reso[1])
    yy, xx = torch.meshgrid((y, x), indexing='ij')
    res = []
    
    coordiantes = torch.stack((xx,yy),dim=-1).reshape(-1,2) + 0.5 #/(torch.FloatTensor(reso[::-1])-1)*2-1
    coordiantes = torch.split(coordiantes,chunk,dim=0)
    for coordiante in tqdm(coordiantes):

        feats,_ = model.get_coding(coordiante.to(model.device))
        y_recon = model.linear_mat(feats)
        
        res.append(y_recon.cpu())
    return torch.cat(res).reshape(reso[0],reso[1],-1)

def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

# %% [markdown]
# # Data loader
# ### please install the av package with "pip install av" if raise a error "pyav:  pip install imageio[pyav]"

# %%
base_conf = OmegaConf.load('../configs/defaults.yaml')
second_conf = OmegaConf.load('../configs/image.yaml')
cfg = OmegaConf.merge(base_conf, second_conf)

device = torch.device('cuda')

HW = [100, 100]

dataset = dataset_dict[cfg.dataset.dataset_name]
tolinear = False if cfg.dataset.datadir.endswith('exr') else True
train_dataset = dataset(cfg.dataset, cfg.training.batch_size, split='train', tolinear=tolinear, HW=HW)
train_loader = DataLoader(train_dataset,
              num_workers=0,
              persistent_workers=False,
              batch_size=None,
              pin_memory=True)

cfg.model.out_dim = train_dataset.img.shape[-1]
batch_size = cfg.training.batch_size
n_iter = cfg.training.n_iters

H,W = train_dataset.HW
cfg.dataset.aabb = train_dataset.scene_bbox

# %%
from models.gausssian2d_3_sh import SplatGaussian2D
import cv2
h, w = HW
model = SplatGaussian2D(n_gaussain_num=3000, n_gaussian_max_pixels=50, n_gaussian_min_pixels=0.5, h=h, w=w)
model.to(device)
print(model)
print('total parameters: ',model.n_parameters())

grad_vars = model.get_optparam_groups(lr_small=0.01)
grad_vars
optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))#



loss_scale = 1.0
lr_factor = 0.1 ** (1 / n_iter)
pbar = tqdm(range(n_iter))
start = time.time()

iter_show = 500
for (iteration, sample) in zip(pbar,train_loader):
    loss_scale *= lr_factor

    coordiantes, pixel_rgb = sample['xy'], sample['rgb']

    coordiantes_device = coordiantes.to(device)

    y_recon = model(coordiantes_device)
    
    loss = torch.mean((y_recon-pixel_rgb.to(device))**2) 

    if iteration % iter_show == 0:
        model.s_max = model.s_max * 0.95
        with torch.no_grad():
            coords_show = train_dataset.coordiante
            coords_show = coords_show.to(device)
            coord_batchmax = 100000
            coord_step = (coords_show.shape[0] + coord_batchmax - 1) // coord_batchmax
            recons = []
            for j in range(coord_step):
                coord_be = j * coord_batchmax
                coord_en = min(coord_be + coord_batchmax, coords_show.shape[0])
                if coord_en <= coord_be:
                    continue
                recons.append(model(coords_show[coord_be:coord_en]))

            recons = torch.cat(recons, dim=0)
            reconsnp = recons.detach().cpu().numpy()
            reconsnp = reconsnp.reshape(HW[0], HW[1], 3)
            reconsnp = cv2.resize(reconsnp, (800, 800))
            gtnp = train_dataset.img.detach().cpu().numpy()
            gtnp = gtnp.reshape(HW[0], HW[1], 3)
            gtnp = cv2.resize(gtnp, (800, 800))
            cv2.imshow("1", reconsnp[:, :, ::-1])
            cv2.imshow("2", gtnp[:, :, ::-1])
            cv2.waitKey(1)
    
    
    psnr = -10.0 * np.log(loss.item()) / np.log(10.0)
    pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' loss_dist = {loss.item():.8f}'
                + f' psnr = {psnr:.3f}'
            )
    
    loss = loss * loss_scale
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
iteration_time = time.time()-start  
    
H,W = train_dataset.HW
img = eval_img(train_dataset.HW).clamp(0,1.)
print(PSNR(img,train_dataset.image.view(img.shape)),iteration_time)
plt.figure(figsize=(10, 10))
plt.imshow(img)

# %%



