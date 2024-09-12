import torch.nn as nn
from networks.model import mana
import argparse


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)

    return network

def get_swin(opts):

    if opts.net_G == 'McMRSR':
        import yaml
        # f = open('/home3/HWGroup/wangcy/JunLyu/lgy/mana_swin_slide/networks/config.yaml')
        f = open('./networks/config.yaml')
        config = yaml.load(f, Loader=yaml.FullLoader)
        network = mana(config, is_training=True)

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters: {}'.format(num_param))
    return set_gpu(network, opts.gpu_ids)
