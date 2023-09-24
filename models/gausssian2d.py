import torch, math
import torch.nn
import torch.nn.functional as F
import numpy as np
import time, skimage
from utils import N_to_reso, N_to_vm_reso


# import BasisCoding

def _build_rotate_mtx(alphas_b):
    # [cos theta, -sintheta]
    # [sin theta,  cos theta]
    costheta = torch.cos(alphas_b)
    sintheta = torch.sin(alphas_b)
    mtx_bx4 = torch.stack([costheta, -sintheta, sintheta, costheta], dim=-1)
    mtx_bx2x2 = mtx_bx4.reshape([*alphas_b.shape, 2, 2])

    return mtx_bx2x2

class SplatGaussian2D(torch.nn.Module):
    def __init__(self,
                 n_gaussain_num,
                 n_gaussian_max_pixels=30,
                 n_gaussian_min_pixels = 0.75,
                 h=512,
                 w=512):
        super().__init__()

        # A 2D gaussian is defined by y = RGB * exp((x - mu)' R'S'SR (x-num))
        # it contains  
        # 1) RGB , 3  
        # 2) mu, 2, 
        # 3) scale, 2, 
        # 4) angle, 1
        # in total 8 params

        # [0, 1], need sigmoid
        rgb_raw = torch.zeros(n_gaussain_num, 3, dtype=torch.float32)

        # [-1.1, 1.1],, need tanh, assume image region is [-1, 1]
        mu_raw = torch.rand(n_gaussain_num, 2, dtype=torch.float32) * 2 - 1
        mu_border = 1.05

        s_min = 1 / n_gaussian_max_pixels
        s_max = 1 / n_gaussian_min_pixels
        # [s_min, s_max], need sigmoid
        scale_raw = torch.zeros(n_gaussain_num, 2, dtype=torch.float32) 

        angle_raw = torch.zeros(n_gaussain_num, dtype=torch.float32) # [-pi, pi]

        self.h = h
        self.w = w
        self.mu_border = mu_border

        self.register_parameter('rgb', torch.nn.Parameter(rgb_raw))
        self.register_parameter('mu', torch.nn.Parameter(mu_raw))

        self.s_min = s_min
        self.s_max = s_max
        self.register_parameter('scale', torch.nn.Parameter(scale_raw))
        self.register_parameter('angle', torch.nn.Parameter(angle_raw))

    def indexing(self,):
        # indexing all gaussians
        pass

    def n_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return total
    
    def get_optparam_groups(self, lr_small=0.01):
        grad_vars = []
        grad_vars += [{'params': self.parameters(), 'lr': lr_small}]
        return grad_vars

    def forward(self, x_bx2, is_train=False):
        
        # given a position x
        # get its corresponding gaussias
        # x, coords, [0.5, w-0.5 or h-0.5]

        b, _ = x_bx2.shape
        dev = x_bx2.device
        wh_2 = torch.Tensor([self.w, self.h]).to(dev)

        # position is x, y
        xnorm_bx2 = x_bx2 / wh_2.reshape(1, 2)
        xnorm_bx2 = xnorm_bx2 * 2 - 1

        mu_nx2 = torch.tanh(self.mu) # [-1, 1]
        mu_nx2 = mu_nx2 * self.mu_border # [-border, border], e.g. [-1.1, 1.1]

        # first, comput the victor from mu to x
        vec_mu_x_bxnx2 = xnorm_bx2.unsqueeze(1) - mu_nx2.unsqueeze(0)
        vec_mu_x_bxnx2x1 = vec_mu_x_bxnx2.unsqueeze(-1)
        vec_mu_x_bxnx2x1 = vec_mu_x_bxnx2x1 / 2 * wh_2.reshape(1, 1, 2, 1)
        
        # second, compute the distance
        # y = A *exp ( x' R' S' S R x )
        alpha_n = torch.tanh(self.angle) * 3.1416 # [-pi, pi]
        alpha_bxn = alpha_n.unsqueeze(0).expand(b, -1)
        R_bxnx2x2 = _build_rotate_mtx(alpha_bxn)

        S_nx2 = torch.sigmoid(self.scale) # [0,1]
        S_nx2 = S_nx2 * (self.s_max - self.s_min) + self.s_min # [s_min, s_max]
        S_bxnx2 = S_nx2.unsqueeze(0).expand(b, -1, -1)
        Sx_bxn, Sy_bxn = S_bxnx2[..., 0], S_bxnx2[..., 1]
        S0_bxn = torch.zeros_like(Sx_bxn)
        S_bxnx4 = torch.stack([Sx_bxn, S0_bxn, S0_bxn, Sy_bxn], dim=-1)
        S_bxnx2x2 = S_bxnx4.reshape(b, -1, 2, 2)

        distance1_bxnx2x1 = S_bxnx2x2 @ R_bxnx2x2 @ vec_mu_x_bxnx2x1
        distance2_bxnx1x1 = distance1_bxnx2x1.permute(0, 1, 3, 2) @ distance1_bxnx2x1

        # thierd, compute the values
        rgb_nx3 = torch.sigmoid(self.rgb)
        rgb_bxnx3 = rgb_nx3.unsqueeze(0).expand(b, -1, -1)

        values_bxnx3 = rgb_bxnx3 * torch.exp(-distance2_bxnx1x1.squeeze(-1))

        values_bx3 = values_bxnx3.sum(dim=1)

        return values_bx3

