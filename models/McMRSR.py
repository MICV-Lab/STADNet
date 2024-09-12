from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from skimage import metrics

from networks import get_swin
from networks.pro_D import gaussian_weights_init
from .utils import AverageMeter, get_scheduler, psnr, DataConsistencyInKspace_I, DataConsistencyInKspace_K, complex_abs_eval, rmse



class RecurrentModel(nn.Module):
    def __init__(self, opts):
        super(RecurrentModel, self).__init__()

        self.loss_names = []
        self.networks = []
        self.optimizers = []

        self.n_recurrent = opts.n_recurrent
        # self.upscale = opts.upscale


        # set default loss flags
        loss_flags = ("w_img_L1")
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G_I = get_swin(opts)
        self.networks.append(self.net_G_I)

        if self.is_train:
            self.loss_names += ['loss_G_L1']
            param = list(self.net_G_I.parameters())
            self.optimizer_G = torch.optim.Adam(param,
                                                lr=opts.lr,
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)
            self.optimizers.append(self.optimizer_G)

        self.criterion = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.opts = opts

        # data consistency layers in image space & k-space
        # dcs_I = []
        # for i in range(self.n_recurrent):
        #     dcs_I.append(DataConsistencyInKspace_I(noise_lvl=None))
        self.dcs_I = DataConsistencyInKspace_I(noise_lvl=None)

        dcs_K = []
        for i in range(self.n_recurrent):
            dcs_K.append(DataConsistencyInKspace_K(noise_lvl=None))
        self.dcs_K = dcs_K

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        self.GT = data['hr'].to(self.device)
        self.ZF = data['lr'].to(self.device)


    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        b, t, c, h, w = self.GT.size()
        ###### data
        ZF = self.ZF
        ZF.requires_grad_(True)
        GT = self.GT
        GT.requires_grad_(True)

        ##### input to model
        PRED = self.net_G_I(ZF) # b t c w h

        ###### dc ######
        PRED = PRED.view(b * t, c, h, w)
        GT = GT.view(b * t, c, h, w)

        self.recon = PRED
        self.gt = GT

    def update_G(self):
        loss_G_L1 = 0
        self.optimizer_G.zero_grad()

        loss_img_l1 = 0
        for j in range(1, self.n_recurrent + 1):
            loss_img_l1 = loss_img_l1 + self.criterion(self.recon, self.gt)


        loss_G_L1 = loss_img_l1
        self.loss_G_L1 = loss_G_L1.item()
        self.loss_img_l1 = loss_img_l1.item()

        total_loss = loss_G_L1
        total_loss.backward()
        self.optimizer_G.step()

    def optimize(self):
        self.loss_G_L1 = 0

        self.forward()
        self.update_G()

    @property
    def loss_summary(self):
        message = ''
        if self.opts.wr_L1 > 0:
            message += 'G_L1: {:.4f} Img_L1: {:.4f} '.format(self.loss_G_L1, self.loss_img_l1)

        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))

    def save(self, filename, epoch, total_iter):

        state = {}
        if self.opts.wr_L1 > 0:
            state['net_G_I'] = self.net_G_I.module.state_dict()
            state['opt_G'] = self.optimizer_G.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)

        if self.opts.wr_L1 > 0:
            self.net_G_I.module.load_state_dict(checkpoint['net_G_I'])
            if train:
                self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    def evaluate(self, loader):
        val_bar = tqdm(loader)
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()
        avg_rmse = AverageMeter()

        recon_images = []
        gt_images = []
        input_images = []

        for data in val_bar:
            self.set_input(data)
            self.forward()
            t=15

            if self.opts.wr_L1 > 0:
                # for i in range(0,t):
                psnr_recon = psnr(complex_abs_eval(self.recon)[0,0,:,:],
                                  complex_abs_eval(self.gt)[0,0,:,:])
                avg_psnr.update(psnr_recon)

                ssim_recon = metrics.structural_similarity(complex_abs_eval(self.recon)[0,0,:,:].cpu().numpy(),
                                  complex_abs_eval(self.gt)[0,0,:,:].cpu().numpy())
                avg_ssim.update(ssim_recon)

                rmse_recon = rmse(complex_abs_eval(self.recon)[0,0,:,:],
                                  complex_abs_eval(self.gt)[0,0,:,:])
                avg_rmse.update(rmse_recon)

                recon_images.append(self.recon.cpu())
                gt_images.append(self.GT.cpu())
                input_images.append(self.ZF.cpu())

            message = 'PSNR: {:4f} '.format(avg_psnr.avg)
            message += 'SSIM: {:4f} '.format(avg_ssim.avg)
            message += 'RMSE: {:8f} '.format(avg_rmse.avg)
            val_bar.set_description(desc=message)

        self.psnr_recon = avg_psnr.avg
        self.ssim_recon = avg_ssim.avg
        self.rmse_recon = avg_rmse.avg

        self.results = {}
        if self.opts.wr_L1 > 0:
            self.results['recon'] = torch.stack(recon_images).squeeze().numpy()
            self.results['gt'] = torch.stack(gt_images).squeeze().numpy()
            self.results['input'] = torch.stack(input_images).squeeze().numpy()