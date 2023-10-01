import torch, math
import torch.nn
import torch.nn.functional as F
import numpy as np
import time, skimage
from utils import N_to_reso, N_to_vm_reso
import nerfacc
from nerfacc import accumulate_along_rays


# import BasisCoding

def _build_rotate_mtx(alphas_b):
    # [cos theta, -sintheta]
    # [sin theta,  cos theta]
    costheta = torch.cos(alphas_b)
    sintheta = torch.sin(alphas_b)
    mtx_bx4 = torch.stack([costheta, -sintheta, sintheta, costheta], dim=-1)
    mtx_bx2x2 = mtx_bx4.reshape([*alphas_b.shape, 2, 2])

    return mtx_bx2x2

def _sh2d(theta_k, shcoef_kxmx3, num_sh):
    
    k, m, _ = shcoef_kxmx3.shape
    assert m == num_sh * 2 + 1

    basis = []
    dc = torch.ones_like(theta_k)
    basis.append(dc)
    for i in range(num_sh):
        sindata = torch.sin(theta_k * (i+1))
        cosdata = torch.cos(theta_k * (i+1))
        basis.append(sindata)
        basis.append(cosdata)
    basis_kxmx1 = torch.stack(basis, dim=-1).unsqueeze(-1)
    rgb = basis_kxmx1 * shcoef_kxmx3
    rgb = rgb.sum(dim=1)

    return rgb




class SplatGaussian2D(torch.nn.Module):
    def __init__(self,
                 n_gaussain_num,
                 n_sh = 4,
                 prune_thres = 5e-4,
                 split_thres = 0.3,
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

        self.num_gaussain = n_gaussain_num
        self.num_sh = n_sh

        self.h = h
        self.w = w

        self.mu_border = 1.05

        s_min = 1 / n_gaussian_max_pixels
        s_max = 1 / n_gaussian_min_pixels

        self.s_min = s_min
        self.s_max = s_max

        self.prune_thres = prune_thres
        self.split_thres = min(split_thres * n_gaussian_max_pixels, 10)

        # [-border, border],, need clip, assume image region is [-1, 1], border is 1.05
        mu_raw = torch.rand(n_gaussain_num, 2, dtype=torch.float32) * 2 - 1
        self.register_parameter('_xyz', torch.nn.Parameter(mu_raw))

        rgb_raw = torch.rand(n_gaussain_num, 2 * n_sh + 1, 3, dtype=torch.float32)
        self.register_parameter('_features', torch.nn.Parameter(rgb_raw))

        # [0, 1], need clip
        opacity_raw = 0.1 + 0.9 * torch.rand(n_gaussain_num, dtype=torch.float32)
        self.register_parameter('_opacity', torch.nn.Parameter(opacity_raw))

        # [0, 1] and rescale to [s_min, s_max], need clip
        scale_raw = torch.rand(n_gaussain_num, 2, dtype=torch.float32) 
        self.register_parameter('_scaling', torch.nn.Parameter(scale_raw))

        # [0, 1] and rescale to [0, 2 * pi]
        angle_raw = torch.rand(n_gaussain_num, dtype=torch.float32)
        self.register_parameter('_rotation', torch.nn.Parameter(angle_raw))

    def indexing(self,):
        # indexing all gaussians
        pass

    @property
    def get_xyz(self):
        gaussian_mu_nx2 = torch.clip(self._xyz, -1, 1) # [-1, 1]
        gaussian_mu_nx2 = gaussian_mu_nx2 * self.mu_border # [-border, border], e.g. [-1.1, 1.1]
        return gaussian_mu_nx2
    
    @property
    def get_scaling(self):
        S_nx2 = torch.clip(self._scaling, 0, 1) # [0,1]
        S_nx2 = S_nx2 * (self.s_max - self.s_min) + self.s_min # [s_min, s_max]
        return S_nx2
    
    @property
    def get_rotation(self):
        angle_n = torch.remainder(self._rotation, 1.0) # 0-1
        angle_n = angle_n * 2 * np.pi
        return angle_n
    
    @property
    def get_features(self):
        features = self._features
        return features
    
    @property
    def get_opacity(self):
        opacity_n = torch.clip(self._opacity, 0, 1)
        return opacity_n
    
    def training_setup(self, lr):

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': lr, "name": "xyz"},
            {'params': [self._features], 'lr': lr, "name": "features"},
            {'params': [self._opacity], 'lr': lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': lr, "name": "rotation"}
        ]

        return l

    def _prune_optimizer(self, mask, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizer, optimizable_tensors

    def prune_points(self, optimizer):
        mask = self._opacity < self.prune_thres
        valid_points_mask = ~mask
        optimizer, optimizable_tensors = self._prune_optimizer(valid_points_mask, optimizer)

        self._xyz = optimizable_tensors["xyz"]
        self._features = optimizable_tensors["features"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]

        self.num_gaussain = valid_points_mask.sum()

        return optimizer
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def reset(self,):
        bad_gaussian = self._opacity < self.prune_thres
        ratio = bad_gaussian.float().mean().detach().cpu().numpy()
        print('%.5f percent of %d agussian has %.5f opacity'%(ratio, self._opacity.shape[0], self.prune_thres))

        large_gaussian = 1.0 / self.get_scaling.min(dim=1)[0] > self.split_thres
        ratio = large_gaussian.float().mean().detach().cpu().numpy()
        print('%.5f percent of %d agussian has %.2f pixels larger scale'%(ratio, self._opacity.shape[0], self.split_thres))
        pass

    def n_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return total
    
    def get_optparam_groups(self, lr_small=0.01):
        grad_vars = []
        grad_vars += [{'params': self.parameters(), 'lr': lr_small}]
        return grad_vars
    
    def distance_compute(self, xnorm_bx2, gaussian_mu_nx2, S_nx2, angle_n, wh_2, rayidxshift):
        dev = xnorm_bx2.device
        n = gaussian_mu_nx2.shape[0]
        bb = xnorm_bx2.shape[0]

        # first, comput the victor from mu to x
        vec_mu_x_bxnx2 = xnorm_bx2.unsqueeze(1) - gaussian_mu_nx2.unsqueeze(0)
        vec_mu_x_bxnx2x1 = vec_mu_x_bxnx2.unsqueeze(-1)
        vec_mu_x_bxnx2x1 = vec_mu_x_bxnx2x1 / 2 * wh_2.reshape(1, 1, 2, 1)

        # rotation will not change distance
        S_bxnx2x1 = S_nx2.reshape(1, -1, 2, 1).expand(bb, -1, -1, -1)
        
        angle_bxn = angle_n.unsqueeze(0).expand(bb, -1)
        R_bxnx2x2 = _build_rotate_mtx(angle_bxn)

        vec_rotate_bxnx2x1 = R_bxnx2x2 @ vec_mu_x_bxnx2x1
        vec_scald_bxnx2x1 = vec_rotate_bxnx2x1 * S_bxnx2x1
        distacce_bxnx1 = vec_scald_bxnx2x1.norm(dim=-2)

        # we only care about close gaussian
        valid_bxn = distacce_bxnx1[..., 0] < 5

        gaussian_index = torch.arange(n, dtype=torch.long).to(dev)
        idx_gaussian_k = gaussian_index.reshape(1, n).expand(bb, -1)[valid_bxn]

        ray_index = torch.arange(bb, dtype=torch.long).to(dev) + rayidxshift
        sum_gaussin_per_ray = valid_bxn.sum(dim=-1)
        idx_ray_k = torch.repeat_interleave(ray_index, sum_gaussin_per_ray, dim=0)

        return idx_gaussian_k, idx_ray_k

    def forward(self, x_bx2, is_train=False):
        
        # given a position x
        # get its corresponding gaussias
        # x, coords, [0.5, w-0.5 or h-0.5]

        b, _ = x_bx2.shape
        dev = x_bx2.device
        wh_2 = torch.Tensor([self.w, self.h]).to(dev)
        n = self.num_gaussain

        # queryed pixels
        # position is x, y
        xnorm_bx2 = x_bx2 / wh_2.reshape(1, 2)
        xnorm_bx2 = xnorm_bx2 * 2 - 1

        # learned gaussians
        gaussian_mu_nx2 = self.get_xyz

        S_nx2 = self.get_scaling

        angle_n = self.get_rotation

        # comppute distance
        # can be acced by cuda
        with torch.no_grad():

            max_pix = 500
            n_step = (b + max_pix - 1) // max_pix
            idx_gaussian_k = []
            idx_ray_k = []
            for i in range(n_step):
                be = i * max_pix
                en = min(be + max_pix, b)
                xnorm_bx2_batch = xnorm_bx2[be:en]
                idx_gaussian_k_batch, idx_ray_k_batch = self.distance_compute(xnorm_bx2_batch, gaussian_mu_nx2, S_nx2, angle_n, wh_2, be)
                idx_gaussian_k.append(idx_gaussian_k_batch)
                idx_ray_k.append(idx_ray_k_batch)
            
            idx_gaussian_k = torch.cat(idx_gaussian_k, dim=0)
            idx_ray_k = torch.cat(idx_ray_k, dim=0)
        
        # compute distance in a compact way
        xnorm_kx2 = xnorm_bx2[idx_ray_k]
        # xnorm_kx2 = torch.repeat_interleave(xnorm_bx2, sum_gaussin_per_ray, dim=0), the same
        mu_kx2 = gaussian_mu_nx2[idx_gaussian_k]
        vec_mu_x_kx2 = xnorm_kx2 - mu_kx2
        vec_mu_x_kx2 = vec_mu_x_kx2 / 2 * wh_2.reshape(1, 2)
        vec_mu_x_kx2x1 = vec_mu_x_kx2.unsqueeze(-1)

        # compute the value
        # y = A *exp ( x' R' S' S R x )
        angle_k = angle_n[idx_gaussian_k]
        R_kx2x2 = _build_rotate_mtx(angle_k)
        S_kx2 = S_nx2[idx_gaussian_k]

        vec_rotate =   R_kx2x2 @ vec_mu_x_kx2x1
        vec_scale_kx2x1 = S_kx2.unsqueeze(-1) * vec_rotate
        distance2_kx1x1 = vec_scale_kx2x1.permute(0, 2, 1) @ vec_scale_kx2x1

        # thierd, compute the values
        vec_mu_x_norm_kx2 = vec_mu_x_kx2 / (1e-10 + vec_mu_x_kx2.norm(dim=-1, keepdim=True))
        vec_mu_x_rot_kx2x1 = R_kx2x2 @ vec_mu_x_norm_kx2.unsqueeze(-1)
        vec_mu_x_rot_kx2 = vec_mu_x_rot_kx2x1.squeeze(-1)

        angle_k = torch.atan2(vec_mu_x_rot_kx2[:, 0], vec_mu_x_rot_kx2[:, 1])
        rgb_sh_nxm = self.get_features
        rgb_sh_kxm = rgb_sh_nxm[idx_gaussian_k]
        rgb_kx3 = _sh2d(angle_k, rgb_sh_kxm, self.num_sh)
        rgb_kx3 = torch.sigmoid(rgb_kx3)

        opacity_n = self.get_opacity
        opacity_k = opacity_n[idx_gaussian_k]
        weights_k = torch.exp(-distance2_kx1x1.reshape(-1,))

        values_bx3 = accumulate_along_rays(weights=weights_k * opacity_k, 
                                           values=rgb_kx3, 
                                           ray_indices=idx_ray_k, 
                                           n_rays=b)

        return values_bx3

