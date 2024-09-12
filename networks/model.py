import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import einsum
from networks.swinir import SwinIR
import math
import warnings
import scipy.io as sio
from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv, SPyNet)
from mmedit.models.common import (ResidualBlockNoBN as ResidualBlockNoBN_,
                                  flow_warp, make_layer as make_layer_)
from einops import rearrange


# Note: residual/upsample blocks are adapted from EDVR: https://github.com/xinntao/BasicSR
def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


#################################
############## new ##############
#################################
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# Feedforward Network
class FeedForward(nn.Module):
    def __init__(self, dim, num_resblocks):
        super().__init__()
        main = []
        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN_, num_resblocks, mid_channels=dim))
        self.net = nn.Sequential(*main)

    def forward(self, x):
        out = self.net(x)
        return out


class FGSW_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(5, 4, 4),
            dim_head=64,
            heads=8,
            shift=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.shift = shift
        inner_dim = dim_head * heads

        # position embedding
        q_l = self.window_size[1] * self.window_size[2]
        kv_l = self.window_size[0] * self.window_size[1] * self.window_size[2]
        self.static_a = nn.Parameter(torch.Tensor(1, heads, q_l, kv_l))
        trunc_normal_(self.static_a)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Conv2d(dim, inner_dim, 3, 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 3, 1, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 3, 1, 1, bias=False)

    def forward(self, q_inp, k_inp, flow):
        """
        :param q_inp: [n,1,c,h,w]
        :param k_inp: [n,2r+1,c,h,w]  (r: temporal radius of neighboring frames)
        :param flow: list: [[n,2,h,w],[n,2,h,w]]
        :return: out: [n,1,c,h,w]
        """
        b, f_q, c, h, w = q_inp.shape
        # print(q_inp.shape)
        fb, hb, wb = self.window_size
        # print(self.window_size)

        [flow_f, flow_b] = flow
        # sliding window
        if self.shift:
            q_inp, k_inp = map(lambda x: torch.roll(x, shifts=(-hb // 2, -wb // 2), dims=(-2, -1)), (q_inp, k_inp))
            if flow_f is not None:
                flow_f = torch.roll(flow_f, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))
            if flow_b is not None:
                flow_b = torch.roll(flow_b, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))
        k_f, k_r, k_b = k_inp[:, 0], k_inp[:, 1], k_inp[:, 2]
        # print(k_f.shape)
        # print(k_r.shape)
        # print(k_b.shape)

        # retrive key elements
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0,
                                                                         w))  # torch.arange(start=1.0,end=6.0)的结果不包括end;  torch.meshgrid（）的功能是生成网格，可以用于生成坐标。
        grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
        grid.requires_grad = False
        grid = grid.type_as(k_f)
        if flow_f is not None:
            vgrid = grid + flow_f.permute(0, 2, 3, 1)
            vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            # index the nearest token
            # k_f = F.grid_sample(k_f.float(), vgrid_scaled, mode='bilinear')
            # 提供一个input的Tensor以及一个对应的flow - field网格(比如光流，体素流等)，然后根据grid中每个位置提供的坐标信息(
            # 这里指input中pixel的坐标)，将input中对应位置的像素值填充到grid指定的位置，得到最终的输出。
            k_f = F.grid_sample(k_f.float(), vgrid_scaled, mode='nearest')
        if flow_b is not None:
            vgrid = grid + flow_b.permute(0, 2, 3, 1)
            vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            # index the nearest token
            # k_b = F.grid_sample(k_b.float(), vgrid_scaled, mode='bilinear')
            k_b = F.grid_sample(k_b.float(), vgrid_scaled, mode='nearest')

        k_inp = torch.stack([k_f, k_r, k_b], dim=1)
        # norm
        q = self.norm_q(q_inp.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        kv = self.norm_kv(k_inp.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        q = self.to_q(q.flatten(0, 1))
        k, v = self.to_kv(kv.flatten(0, 1)).chunk(2, dim=1)
        # print(q.shape,k.shape,v.shape) #torch.Size([2, 512, 64, 64]) torch.Size([6, 512, 64, 64]) torch.Size([6, 512, 64, 64])

        # split into (B,N,C)
        q, k, v = map(lambda t: rearrange(t, '(b f) c (h p1) (w p2)-> (b h w) (f p1 p2) c', p1=hb, p2=wb, b=b),
                      (q, k, v))

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        # print(q.shape, k.shape, v.shape)

        # scale
        q *= self.scale
        # print(q.shape)
        # attention
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        # print(sim.shape,self.static_a.shape)
        sim = sim + self.static_a
        attn = sim.softmax(dim=-1)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')

        # merge windows back to original feature map
        out = rearrange(out, '(b h w) (f p1 p2) c -> (b f) c (h p1) (w p2)', b=b, h=(h // hb), w=(w // wb),
                        p1=hb, p2=wb)

        # combine heads
        out = self.to_out(out).view(b, f_q, c, h, w)

        # inverse shift
        if self.shift:
            out = torch.roll(out, shifts=(hb // 2, wb // 2), dims=(-2, -1))

        return out


class FGAB(nn.Module):
    def __init__(
            self,
            q_dim,
            emb_dim,
            window_size=(3, 4, 4),
            dim_head=64,
            heads=8,
            num_resblocks=5,
            shift=False
    ):
        super().__init__()
        self.window_size = window_size
        self.heads = heads
        self.embed_dim = emb_dim
        self.q_dim = q_dim
        self.attn = FGSW_MSA(q_dim, window_size, dim_head, heads, shift=shift)
        self.feed_forward = FeedForward(q_dim, num_resblocks)
        self.conv = nn.Conv2d(q_dim + emb_dim, q_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.shift = shift

    def forward(self, x, flows_forward, flows_backward):
        """
        :param x: [n,t,c,h,w]
        :param flows_forward: [n,t,2,h,w]
        :param flows_backward: [n,t,2,h,w]
        :return: outs: [n,t,c,h,w]
        """
        x = x.permute(1, 0, 2, 3, 4)
        t, n, c, h, w = x.shape
        # print(x.shape)
        outs = []
        embedding = flows_forward[0].new_zeros(n, self.embed_dim, h, w)
        for i in range(0, t):
            flow_f, flow_b = None, None
            if i > 0:
                flow_f = flows_forward[i - 1]
                embedding = flow_warp(embedding, flow_f.permute(0, 2, 3, 1))
                k_f = x[i - 1]
            else:
                k_f = x[i]
            if i < t - 1:
                flow_b = flows_backward[i]
                k_b = x[i + 1]
            else:
                k_b = x[i]
            x_current = x[i]

            # print(embedding.shape)
            # print(x_current.shape)
            q_inp = self.lrelu(self.conv(torch.cat((embedding, x_current), dim=1))).unsqueeze(1)
            k_inp = torch.stack([k_f, x_current, k_b], dim=1)
            out = self.attn(q_inp=q_inp, k_inp=k_inp, flow=[flow_f, flow_b]) + q_inp
            out = out.squeeze(1)
            out = self.feed_forward(out) + out
            embedding = out

            outs.append(out)
        return outs


class nonlocal_attention(nn.Module):
    def __init__(self, config, is_training=True):
        super(nonlocal_attention, self).__init__()
        self.in_channels = config['in_channels']
        self.inter_channels = self.in_channels // 2
        self.is_training = is_training
        width = config['width']
        height = config['height']

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Conv2d(in_channels=self.inter_channels * 3, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W_z.weight, 0)
        nn.init.constant_(self.W_z.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        self.norm = torch.nn.GroupNorm(num_groups=30, num_channels=self.in_channels, eps=1e-6, affine=True)

        x1 = np.linspace(0, width - 1, width)
        y1 = np.linspace(0, height - 1, height)
        x2 = np.linspace(0, width - 1, width)
        y2 = np.linspace(0, height - 1, height)
        X1, Y1, Y2, X2 = np.meshgrid(x1, y1, y2, x2)
        D = (X1 - X2) ** 2 + (Y1 - Y2) ** 2
        D = torch.from_numpy(D)
        D = rearrange(D, 'a b c d -> (a b) (c d)')
        if self.is_training:
            D = D.float()
        else:
            D = D.half()
        self.D = torch.nn.Parameter(D, requires_grad=False)
        self.std = torch.nn.Parameter(4 * torch.ones(1).float())
        if self.is_training == False:
            self.std = self.std.half()
        self.W_z1 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W_z1.weight, 0)
        nn.init.constant_(self.W_z1.bias, 0)
        # self.mb = torch.nn.Parameter(torch.randn(self.inter_channels, 256))

    def forward(self, x, i):
        b, t, c, h, w = x.size()
        if i == 1:
            outputs_z = []
            for j in range(0, 2):
                q = x[:, j, :, :, :]
                weight = torch.exp(-0.5 * (self.D / (self.std * self.std)))
                weight = weight.unsqueeze(0).repeat(b, 1, 1)
                reshaped_x = x.view(b * t, c, h, w).contiguous()
                h_ = self.norm(reshaped_x)
                q_ = self.norm(q)
                g_x = self.g(h_).view(b, t, self.inter_channels, h, w).contiguous()
                theta_x = self.theta(h_).view(b, t, self.inter_channels, h, w).contiguous()
                phi_x = self.phi(q_).view(b, self.inter_channels, -1)
                phi_x = phi_x.permute(0, 2, 1).contiguous()
                corr_l = []
                for i in range(t):
                    theta = theta_x[:, i, :, :, :]
                    g = g_x[:, i, :, :, :]

                    g = g.view(b, self.inter_channels, -1).permute(0, 2, 1).contiguous()
                    theta = theta.view(b, self.inter_channels, -1).contiguous()

                    if self.is_training:
                        f = torch.matmul(phi_x, theta)
                    else:
                        f = torch.matmul(phi_x.half(), theta.half())

                    f_div_C = F.softmax(f, dim=-1) * weight
                    if self.is_training:
                        y = torch.matmul(f_div_C, g).float()
                    else:
                        y = torch.matmul(f_div_C, g.half()).float()
                    y = y.permute(0, 2, 1).view(b, self.inter_channels, h, w)
                    corr_l.append(y)

                corr_prob = torch.cat(corr_l, dim=1).view(b, -1, h, w)
                W_y = self.W_z(corr_prob)
                z = W_y + q

                outputs_z.append(z)
            z = torch.stack(outputs_z, dim=1)
        elif i == 13:
            outputs_z = []
            for j in range(1, 3):
                q = x[:, j, :, :, :]
                weight = torch.exp(-0.5 * (self.D / (self.std * self.std)))
                weight = weight.unsqueeze(0).repeat(b, 1, 1)
                reshaped_x = x.view(b * t, c, h, w).contiguous()
                h_ = self.norm(reshaped_x)
                q_ = self.norm(q)
                g_x = self.g(h_).view(b, t, self.inter_channels, h, w).contiguous()
                theta_x = self.theta(h_).view(b, t, self.inter_channels, h, w).contiguous()
                phi_x = self.phi(q_).view(b, self.inter_channels, -1)
                phi_x = phi_x.permute(0, 2, 1).contiguous()
                corr_l = []
                for i in range(t):
                    theta = theta_x[:, i, :, :, :]
                    g = g_x[:, i, :, :, :]

                    g = g.view(b, self.inter_channels, -1).permute(0, 2, 1).contiguous()
                    theta = theta.view(b, self.inter_channels, -1).contiguous()

                    if self.is_training:
                        f = torch.matmul(phi_x, theta)
                    else:
                        f = torch.matmul(phi_x.half(), theta.half())

                    f_div_C = F.softmax(f, dim=-1) * weight
                    if self.is_training:
                        y = torch.matmul(f_div_C, g).float()
                    else:
                        y = torch.matmul(f_div_C, g.half()).float()
                    y = y.permute(0, 2, 1).view(b, self.inter_channels, h, w)
                    corr_l.append(y)

                corr_prob = torch.cat(corr_l, dim=1).view(b, -1, h, w)
                W_y = self.W_z(corr_prob)
                z = W_y + q

                outputs_z.append(z)
            z = torch.stack(outputs_z, dim=1)
        else:
            q = x[:, 1, :, :, :]

            weight = torch.exp(-0.5 * (self.D / (self.std * self.std)))
            weight = weight.unsqueeze(0).repeat(b, 1, 1)

            reshaped_x = x.view(b * t, c, h, w).contiguous()
            h_ = self.norm(reshaped_x)
            q_ = self.norm(q)

            g_x = self.g(h_).view(b, t, self.inter_channels, h, w).contiguous()
            theta_x = self.theta(h_).view(b, t, self.inter_channels, h, w).contiguous()
            phi_x = self.phi(q_).view(b, self.inter_channels, -1)
            phi_x = phi_x.permute(0, 2, 1).contiguous()

            corr_l = []
            for i in range(t):
                theta = theta_x[:, i, :, :, :]
                g = g_x[:, i, :, :, :]

                g = g.view(b, self.inter_channels, -1).permute(0, 2, 1).contiguous()
                theta = theta.view(b, self.inter_channels, -1).contiguous()

                if self.is_training:
                    f = torch.matmul(phi_x, theta)
                else:
                    f = torch.matmul(phi_x.half(), theta.half())

                f_div_C = F.softmax(f, dim=-1) * weight
                if self.is_training:
                    y = torch.matmul(f_div_C, g).float()
                else:
                    y = torch.matmul(f_div_C, g.half()).float()
                y = y.permute(0, 2, 1).view(b, self.inter_channels, h, w)
                corr_l.append(y)

            corr_prob = torch.cat(corr_l, dim=1).view(b, -1, h, w)
            W_y = self.W_z(corr_prob)

            z = W_y + q

        return z


class mana(nn.Module):
    def __init__(self, config, is_training):
        super(mana, self).__init__()
        self.in_channels = config['in_channels']
        self.conv_first = nn.Conv2d(1, self.in_channels, 3, 1, 1)

        self.swin = SwinIR(upscale=config['upscale'], img_size=(config['width'], config['height']),
                         window_size=config['window_size'], img_range=1., depths=[6, 6, 6, 6],
                         embed_dim=config['in_channels'], num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')


        self.nonloc_spatial = nonlocal_attention(config, is_training)
        self.cat_conv = nn.Conv2d(self.in_channels * 2, self.in_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.window_size = [3, 3, 3]

        #### optical flow
        self.spynet = SPyNet(pretrained=None)
        self.fgab = FGAB(
            q_dim=self.in_channels, emb_dim=self.in_channels, window_size=self.window_size, heads=4,
            dim_head=self.in_channels,
            num_resblocks=5, shift=[False]
        )

        # upsample
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(self.in_channels, 64, 3, 1, 1),
                                                  nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(4, 64)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)

    def spatial_padding(self, lqs):

        n, t, c, h, w = lqs.shape
        tb, hb, wb = self.window_size
        hb *= 4
        wb *= 4
        pad_h = (hb - h % hb) % hb
        pad_w = (wb - w % wb) % wb

        # padding
        lqs = lqs.view(-1, c, h, w)
        lqs = F.pad(lqs, [0, pad_w, 0, pad_h], mode='reflect')

        return lqs.view(n, t, c, h + pad_h, w + pad_w)

    def compute_flow(self, lqs, flows):
        n, t, c, h, w = lqs.size()
        flows['forward'], flows['backward'] = [], []
        lqs_1 = lqs[:, :-1, :, :, :]
        lqs_2 = lqs[:, 1:, :, :, :]
        for i in range(t - 1):
            lq_1, lq_2 = lqs_1[:, i, :, :, :], lqs_2[:, i, :, :, :]
            flow_backward_ = self.spynet(lq_1, lq_2)
            flow_forward_ = self.spynet(lq_2, lq_1)

            flows['forward'].append(flow_forward_)
            flows['backward'].append(flow_backward_)

        # #####save######
        # flows_results = {}
        # flows_results['forward'] = torch.stack(flows['forward']).squeeze().cpu().detach().numpy()
        # flows_results['backward'] = torch.stack(flows['backward']).squeeze().cpu().detach().numpy()
        #
        # sio.savemat('flows.mat', flows_results)
        return flows

    def forward(self, inp):
        # Conv2 = nn.Conv2d(self.in_channels*2, self.in_channels, kernel_size=3, padding=0, stride=1, bias=True).to(device)
        _, frame_t, _, _, _ = inp.size()
        # print(frame_t)
        fin_out_begin = []
        fin_out_body = []
        fin_out_end = []
        for i in range(1, frame_t - 1):
            # print(i)
            if i == 1:
                # print("this is frame begin")
                x = inp[:, i - 1:i + 2, :, :, :].contiguous()
                b, t, c, h, w = x.size()
                res_x = x
                ########## spatial_padding #######
                lqs = self.spatial_padding(x)
                h_, w_ = lqs.size(3), lqs.size(4)
                ############# encoder ############
                lqs_encode = self.swin(lqs.view(-1, c, h_, w_), encoder=True)
                lqs_mem = lqs_encode.view(b, t, self.in_channels, h_, w_).contiguous()
                ########## flows brach ###########
                flows = {}
                flows = self.compute_flow(lqs, flows)
                flow_out = self.fgab(lqs_mem, flows_forward=flows['forward'], flows_backward=flows['backward'])
                flow_out = torch.stack(flow_out, dim=1)
                ########## non-local attention brach ##########
                res = self.nonloc_spatial(lqs_mem[:, :, :, :h, :w], i)
                ########## sum ###########
                flow_out = flow_out[:, :, :, :h, :w]
                # res = res + flow_out[:, 0:2, :, :, :]
                ######### cat  #########
                res_flow = torch.cat((res, flow_out[:, 0:2, :, :, :]), dim=2) # ( bs, t, in_channels*2, 32, 32)
                cat_out = []
                for i in range(0, 2):
                    d_dim = self.cat_conv(res_flow[:, i, :, :, :])
                    cat_out.append(d_dim)  # ( bs, in_channels, 32 ,32)
                res = torch.stack(cat_out, dim=1)
                ######### decoder  #########
                out = self.swin(res.view(-1, self.in_channels, h, w), encoder=False).view(b, 2, self.in_channels, h, w).contiguous()
                ######### upsample  #########
                for i in range(0, 2):
                    out_ = out[:, i, :, :, :]
                    out_ = self.conv_before_upsample(out_)
                    out_ = self.conv_last(self.upsample(out_))
                    lx = F.interpolate(res_x[:, i, :, :, :], scale_factor=4, mode='bilinear', align_corners=False)
                    oup_ = lx + out_
                    fin_out_begin.append(oup_)
                oup_begin = torch.stack(fin_out_begin, dim=1)
                # print("this is frame begin end")
            elif i == frame_t - 2:
                # print("this is frame end")
                x = inp[:, i - 1:i + 2, :, :, :].contiguous()
                b, t, c, h, w = x.size()
                res_x = x
                ########## spatial_padding #######
                lqs = self.spatial_padding(x)
                h_, w_ = lqs.size(3), lqs.size(4)
                ############# encoder ############
                lqs_encode =  self.swin(lqs.view(-1, c, h_, w_), encoder=True)
                lqs_mem = lqs_encode.view(b, t, self.in_channels, h_, w_).contiguous()
                ########## flows brach ###########
                flows = {}
                flows = self.compute_flow(lqs, flows)
                flow_out = self.fgab(lqs_mem, flows_forward=flows['forward'], flows_backward=flows['backward'])
                flow_out = torch.stack(flow_out, dim=1)
                ########## non-local attention brach ##########
                res = self.nonloc_spatial(lqs_mem[:, :, :, :h, :w], i)
                ########## sum ###########
                flow_out = flow_out[:, :, :, :h, :w]
                # res = res + flow_out[:, 1:3, :, :, :]
                ######### cat  #########
                cat_out = []
                res_flow = torch.cat((res, flow_out[:, 1:3, :, :, :]), dim=2)  # ( bs, t, in_channels*2, 32, 32)
                for i in range(0, 2):
                    d_dim = self.cat_conv(res_flow[:, i, :, :, :])
                    cat_out.append(d_dim)  # ( bs, in_channels, 32 ,32)
                res = torch.stack(cat_out, dim=1)
                ######### decoder  #########
                out = self.swin(res.view(-1, self.in_channels, h, w), encoder=False).view(b, 2, self.in_channels, h, w).contiguous()
                ######### upsample  #########
                for i in range(0, 2):
                    out_ = out[:, i, :, :, :]
                    out_ = self.conv_before_upsample(out_)
                    out_ = self.conv_last(self.upsample(out_))
                    lx = F.interpolate(res_x[:, i, :, :, :], scale_factor=4, mode='bilinear', align_corners=False)
                    oup_ = lx + out_
                    fin_out_end.append(oup_)
                oup_end = torch.stack(fin_out_end, dim=1)
                # print("this is frame end end")
            else:
                # print("this is frame body")
                x = inp[:, i - 1:i + 2, :, :, :].contiguous()
                b, t, c, h, w = x.size()
                lx = F.interpolate(x[:, 1, :, :, :], scale_factor=4, mode='bilinear', align_corners=False)
                ########## spatial_padding #######
                lqs = self.spatial_padding(x)
                h_, w_ = lqs.size(3), lqs.size(4)
                ############# encoder ############
                lqs_encode = self.swin(lqs.view(-1, c, h_, w_), encoder=True)
                lqs_mem = lqs_encode.view(b, t, self.in_channels, h_, w_).contiguous()
                ########## flows brach ###########
                flows = {}
                flows = self.compute_flow(lqs, flows)
                flow_out = self.fgab(lqs_mem, flows_forward=flows['forward'], flows_backward=flows['backward'])
                flow_out = torch.stack(flow_out, dim=1)
                ########## non-local attention brach ##########
                res = self.nonloc_spatial(lqs_mem[:, :, :, :h, :w], i)
                ########## sum ###########
                flow_out = flow_out[:, :, :, :h, :w]
                # res = res + flow_out[:, 1, :, :, :]
                ######### cat  #########
                # [2, 120, 32, 32]
                res_flow = torch.cat((res, flow_out[:, 1, :, :, :]), dim=1)  # ( bs, in_channels*2, 32, 32)
                d_dim = self.cat_conv(res_flow[:, :, :, :])  # ( bs, in_channels, 32 ,32)
                res = d_dim
                #res2 = torch.cat((res1[:, :, :, :], flow_out[:, 1, :, :, :]), dim=1)  # ( bs, in_channels*2 , 32 ,32)
                #res[:, :, :, :] = Conv2(res2).view(b, self.in_channels, h, w).contiguous()  # ( bs, in_channels, 32 ,32)
                ######### decoder  #########
                out = self.swin(res.view(-1, self.in_channels, h, w), encoder=False)
                ######### upsample  #########
                out = self.conv_before_upsample(out)
                out = self.upsample(out)
                print_out = out.cpu().detach().numpy()
                np.save('print_out.npy', print_out)
                out = self.conv_last(out)
                out = lx + out
                fin_out_body.append(out)
                # print("this is frame body end")
        oup_body = torch.stack(fin_out_body, dim=1)

        # print(oup_begin.shape,oup_body.shape,oup_end.shape)
        fin_oup = torch.cat([oup_begin, oup_body, oup_end], dim=1)

        return fin_oup


if __name__ == '__main__':
    import yaml

    f = open('config.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)

    model = mana(config, is_training=True)
    num_param = sum([p.numel() for p in model.parameters() if p.requires_grad])
    # print(model)
    print('Number of parameters: {}'.format(num_param))

    inp = torch.randn(1, 15, 1, 32, 32)

    out = model(inp)
    print(out.shape)
