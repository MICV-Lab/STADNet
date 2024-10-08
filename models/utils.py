import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log10, sqrt
from torch.optim import lr_scheduler
import scipy.io as sio
import pdb
import torch.fft as FFT


def get_nonlinearity(name):
    """Helper function to get non linearity module, choose from relu/softplus/swish/lrelu"""
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'swish':
        return Swish(inplace=True)
    elif name == 'lrelu':
        return nn.LeakyReLU()


class Swish(nn.Module):
    def __init__(self, inplace=False):
        """The Swish non linearity function"""
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_scheduler(optimizer, opts, last_epoch=-1):
    if 'lr_policy' not in opts or opts.lr_policy == 'constant':
        scheduler = None
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.step_size,
                                        gamma=opts.gamma, last_epoch=last_epoch)
    elif opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.epoch_decay) / float(opts.n_epochs - opts.epoch_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)
    return scheduler


def get_recon_loss(opts):
    loss = None
    if opts['recon'] == 'L2':
        loss = nn.MSELoss()
    elif opts['recon'] == 'L1':
        loss = nn.L1Loss()

    return loss


def psnr(sr_image, gt_image):
    # print(sr_image.shape, gt_image.shape)
    # assert sr_image.size(0) == gt_image.size(0) == 1

    peak_signal = (gt_image.max() - gt_image.min()).item()

    mse = (sr_image - gt_image).pow(2).mean().item()

    return 10 * log10(peak_signal ** 2 / mse)


def mse(sr_image, gt_image):
    # assert sr_image.size(0) == gt_image.size(0) == 1

    mse = (sr_image - gt_image).pow(2).mean().item()

    return mse

def rmse(sr_image, gt_image):
    # assert sr_image.size(0) == gt_image.size(0) == 1

    mse = (sr_image - gt_image).pow(2).mean().item()
    rmse = sqrt(mse)

    return rmse


'''
K-Space
'''
def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space [b,w,h,2] need to [b,2,w,h]
    k0   - initially sampled elements in k-space
    dc_mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


class DataConsistencyInKspace_I(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None):
        super(DataConsistencyInKspace_I, self).__init__()
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, pred, gt, dc_mask):
        """
            pred - [n*t,c,h,w] need to [n*t,h,w,c] [n,2,w,h] -> [n,w,h,2]
            zf   - [n*t,c,h,w] need to [n*t,h,w,c] [n,2,w,h] -> [n,w,h,2]
            mask - [n*t,c,h,w] need to [n*t,h,w,c] [n,2,w,h] -> [n,w,h,2]
            """
        pred = pred.permute(0, 2, 3, 1).contiguous() # [n,w,h,2]
        gt = gt.permute(0, 2, 3, 1).contiguous() # [n,w,h,2]
        dc_mask = dc_mask.permute(0, 2, 3, 1).contiguous() # [n,w,h,2]


        #-----to compelx  [w,h]
        gt = torch.view_as_complex(gt)  # [w,h]
        gt_k = FFT2D(gt)
        pred = torch.view_as_complex(pred) #[w,h]
        pred_k = FFT2D(pred)
        #-----to 2 channel
        gt_k = torch.view_as_real(gt_k)  # [n,w,h,2]
        pred_k = torch.view_as_real(pred_k)#[n,w,h,2]

        out_k = data_consistency(pred_k,gt_k,dc_mask) #[n,w,h,2]

        #-------to compelx
        out_k_ = torch.view_as_complex(out_k)
        out_dc = IFFT2D(out_k_) # compelx
        out_dc = torch.view_as_real(out_dc).contiguous()

        #------
        out_k = out_k.permute(0, 3, 1, 2).contiguous() # to [n,2,w,h]
        out_dc = out_dc.permute(0, 3, 1, 2).contiguous()  # to [n,2,w,h]
        gt_k = gt_k.permute(0, 3, 1, 2).contiguous() # to [n,2,w,h]

        return out_k, out_dc, gt_k


class DataConsistencyInKspace_K(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None):
        super(DataConsistencyInKspace_K, self).__init__()
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, k, k0, mask):
        """
        k    - input in frequency domain, of shape (n, 2, nx, ny)
        k0   - initially sampled elements in k-space
        dc_mask - corresponding nonzero location
        """

        if k.dim() == 4:  # input is 2D [b,2,w,h]
            k = k.permute(0, 2, 3, 1) #[b,w,h,2]
        else:
            raise ValueError("error in data consistency layer!")

        out = data_consistency(k, k0, mask, self.noise_lvl) #[b,2,w,h]
        x_res = IFFT2D(out) #[b,2,w,h]
        # ========
        # ks_net_fin_out = x_res.cpu().detach().numpy()
        # sio.savemat('ks_net_fin_out.mat', {'data': ks_net_fin_out});
        # ========

        if k.dim() == 4:
            # x_res = x_res.permute(0, 3, 1, 2)
            x_res = x_res
        else:
            raise ValueError("Iuput dimension is wrong, it has to be a 2D input!")

        return x_res, out

# Basic functions / transforms
def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def FFT2D(x):
    return FFT.fftshift(FFT.fft2(x, dim=(-2, -1)))


def IFFT2D(x):
    return FFT.ifft2(FFT.ifftshift(x), dim=(-2, -1))


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def complex_abs_eval(data):
    # assert data.size(1) == 2
    # return (data[:, 0:1, :, :] ** 2 + data[:, 1:2, :, :] ** 2).sqrt()
    return data


def to_spectral_img(data):
    """
    Compute the spectral images of a kspace data
    with keeping each column for creation of one spectral image
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2

    spectral_vol = torch.zeros([data.size(-2), data.size(-2), data.size(-2)])

    for i in range(data.size(-2)):
        kspc1 = torch.zeros(data.size())
        kspc1[:, i, :] = data[:, i, :]
        img1 = IFFT2D(kspc1)
        img1_abs = complex_abs(img1)

        spectral_vol[i, :, :] = img1_abs

    return spectral_vol