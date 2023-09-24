import torch.nn
import torch.nn.functional as F


def Fgrid_sample_nearest(data_bxcxhxw, grid_bxhxwx2, align_corners):

    b, c, h, w = data_bxcxhxw.shape
    if align_corners == False:
        # u = (x + 0.5) / w * 2 - 1
        x_bxhxw = (grid_bxhxwx2[..., 0] + 1) / 2 * w  - 0.5
        y_bxhxw = (grid_bxhxwx2[..., 1] + 1) / 2 * h - 0.5
    else:
        # u0 & x0 are aligned
        # when x is -1 --> u[0] -> 0.5
        # when x is 1 -->  u[w-1] -> w-0.5
        x_bxhxw = (grid_bxhxwx2[..., 0] + 1) / 2 * (w-1) + 0.5 
        y_bxhxw = (grid_bxhxwx2[..., 1] + 1) / 2 * (w-1) + 0.5 
    
    x_bxhxw = x_bxhxw + 0.5
    y_bxhxw = y_bxhxw + 0.5
    x_near = torch.floor(x_bxhxw)
    y_near = torch.floor(y_bxhxw)

    x_near_long = torch.clip(x_near, 0, w-1).long()
    y_near_long = torch.clip(y_near, 0, h-1).long()

    batch_shift = torch.arange(b).to(grid_bxhxwx2.device).long()
    batch_shift_bxhxw = batch_shift.reshape(-1, 1, 1).repeat(1, grid_bxhxwx2.shape[1], grid_bxhxwx2.shape[2])
    batch_shift_bxhxw = batch_shift_bxhxw * grid_bxhxwx2.shape[1] * grid_bxhxwx2.shape[2]

    index_bxy = y_near_long * w + x_near_long + batch_shift_bxhxw

    data_bhwxc = data_bxcxhxw.permute(0, 2, 3, 1).reshape(-1, c)

    d1 = data_bhwxc[index_bxy.reshape(-1,)]
    d_bhwxc = d1

    b, h, w, _ = grid_bxhxwx2.shape
    d_bxhxwxc = d_bhwxc.reshape(b, h, w, -1)
    d_bxcxhxw = d_bxhxwxc.permute(0, 3, 1, 2)

    return d_bxcxhxw



def Fgraid_sample(data_bxcxhxw, grid_bxhxwx2, mode='bilinear', align_corners=False, padding_mode='border'):
    
    if mode == 'nearest':
        return Fgrid_sample_nearest(data_bxcxhxw, grid_bxhxwx2, align_corners)
    
    assert mode == 'bilinear'
    
    b, c, h, w = data_bxcxhxw.shape
    if align_corners == False:
        # u = (x + 0.5) / w * 2 - 1
        x_bxhxw = (grid_bxhxwx2[..., 0] + 1) / 2 * w  - 0.5
        y_bxhxw = (grid_bxhxwx2[..., 1] + 1) / 2 * h - 0.5
    else:
        # u0 & x0 are aligned
        # when x is -1 --> u[0] -> 0.5
        # when x is 1 -->  u[w-1] -> w-0.5
        x_bxhxw = (grid_bxhxwx2[..., 0] + 1) / 2 * (w-1) + 0.5 
        y_bxhxw = (grid_bxhxwx2[..., 1] + 1) / 2 * (w-1) + 0.5 
    
    x_left = torch.floor(x_bxhxw)
    x_right = torch.ceil(x_bxhxw)
    w_x_left = x_right - x_bxhxw
    w_x_right = x_bxhxw - x_left

    y_left = torch.floor(y_bxhxw)
    y_right = torch.ceil(y_bxhxw)
    w_y_left = y_right - x_bxhxw
    w_y_right = y_bxhxw - y_left

    x_left_long = torch.clip(x_left, 0, w-1).long()
    x_right_long = torch.clip(x_right, 0, w-1).long()

    y_left_long = torch.clip(y_left, 0, h-1).long()
    y_right_long = torch.clip(y_right, 0, h-1).long()

    batch_shift = torch.arange(b).to(grid_bxhxwx2.device).long()
    batch_shift_bxhxw = batch_shift.reshape(-1, 1, 1).repeat(1, h, w)
    batch_shift_bxhxw = batch_shift_bxhxw * h * w

    index_x_left_y_left = y_left_long * w + x_left_long + batch_shift_bxhxw
    index_x_left_y_right = y_right_long * w + x_left_long + batch_shift_bxhxw
    index_x_right_y_left = y_left_long * w + x_right_long + batch_shift_bxhxw
    index_x_right_y_right = y_right_long * w + x_right_long + batch_shift_bxhxw

    w_x_left_y_left = w_x_left * w_y_left
    w_x_left_y_right = w_x_left * w_y_right
    w_x_right_y_left = w_x_right * w_y_left
    w_x_right_y_right = w_x_right * w_y_right

    data_bhwxc = data_bxcxhxw.permute(0, 2, 3, 1).reshape(-1, c)

    d1 = data_bhwxc[index_x_left_y_left.reshape(-1,)] * w_x_left_y_left
    d2 = data_bhwxc[index_x_left_y_right.reshape(-1,)] * w_x_left_y_right
    d3 = data_bhwxc[index_x_right_y_left.reshape(-1,)] * w_x_right_y_left
    d4 = data_bhwxc[index_x_right_y_right.reshape(-1,)] * w_x_right_y_right

    d_bhwxc = d1 + d2 + d3 + d4

    b, h, w, _ = grid_bxhxwx2.shape
    d_bxhxwxc = d_bhwxc.reshape(b, h, w, -1)
    d_bxcxhxw = d_bxhxwxc.permute(0, 3, 1, 2)

    return d_bxcxhxw




