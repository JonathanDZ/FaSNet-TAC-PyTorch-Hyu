# -*- coding: utf-8 -*-
"""
Created on 2018/12
Author: Kaituo XU

Edited by: yoonsanghyu  2020/04

"""

# # @BJ Single card training
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='1'
# showing NCCL debugging information
# import os
# os.environ["NCCL_DEBUG"] = "INFO"

import argparse
import torch

from data import AudioDataset, AudioDataLoader, AdhocDataset, AdhocDataLoader, AdhocDataset2, AdhocDataLoader2

from FaSNet import FaSNet_TAC
from solver import Solver

# @BJ import iFasnet model
from iFaSNet import iFaSNet
from utility.TFGridNet import make_TF_GridNet

# #BJ Multi-GPU training with DDP
# import all essential modules
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

# $BJ a helper to initialize the distributed process group
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        word_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

parser = argparse.ArgumentParser( "FaSNet + TAC model")

# General config
# Task related
parser.add_argument('--tr_json', type=str, default=None, help='path to tr.json')
parser.add_argument('--cv_json', type=str, default=None, help='path to cv.json')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--segment', default=4, type=float, help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=20, type=float, help='max audio length (seconds) in cv, to avoid OOM issue.')

# Network architecture
parser.add_argument('--enc_dim', default=64, type=int, help='Number of filters in autoencoder')
parser.add_argument('--win_len', default=4, type=int, help='Number of convolutional blocks in each repeat') # fasnet:4, ifasnet:16
parser.add_argument('--context_len', default=16, type=int, help='context window size')
parser.add_argument('--feature_dim', default=64, type=int, help='feature dimesion')
parser.add_argument('--hidden_dim', default=128, type=int, help='Hidden dimension')
parser.add_argument('--layer', default=4, type=int, help='Number of layer in dprnn step') # fasnet:4, ifasnet:6
parser.add_argument('--segment_size', default=50, type=int, help="segment_size") # fasnet:50, ifasnet:24
parser.add_argument('--nspk', default=2, type=int, help='Maximum number of speakers')
parser.add_argument('--mic', default=6, type=int, help='number of microphone')

# Training config
parser.add_argument('--use_cuda', type=int, default=1, help='Whether use GPU')
parser.add_argument('--epochs', default=120, type=int, help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=1, type=int, help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=1, type=int, help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=1, type=float, help='Gradient norm threshold to clip') #fasnet:5 TFGridNet:1


# minibatch
parser.add_argument('--shuffle', default=1, type=int, help='reshuffle the data at every epoch')
parser.add_argument('--drop', default=0, type=int, help='drop files shorter than this')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to generate minibatch')


# optimizer
parser.add_argument('--optimizer', default='adam', type=str, choices=['sgd', 'adam'], help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float, help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float, help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float, help='weight decay (L2 penalty)')


# save and load model
parser.add_argument('--save_folder', default='exp/tmp', help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int, help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--tseed', default=-1, help='Torch random seed', type=int)
parser.add_argument('--nseed', default=-1, help='Numpy random seed', type=int)


# logging
parser.add_argument('--print_freq', default=1000, type=int, help='Frequency of printing training infomation')


# @BJ switch between adhoc and fixed array configuration
parser.add_argument('--array_type', default="fixed", type=str, choices=["fixed", "adhoc"],
                    help='Enable one of the training mode, either can be "fixed" or "adhoc"')

# @BJ change to iFasnet model
parser.add_argument('--model', default= "TF-GridNet", type=str, choices=["ifasnet", "fasnet", "TF-GridNet"],
                    help="choose different model for training")


# @BJ change function name to train_worker, since it only handles training on a single device
def train_worker(rank, world_size, args):

    
    ddp_setup(rank, world_size)
    # data
    # @BJ Switch between fixed mode and adhoc mode
    # if args.array_type == "fixed":
    #     tr_dataset = AudioDataset('tr', batch_size = args.batch_size, sample_rate= args.sample_rate, nmic = args.mic)
    #     cv_dataset = AudioDataset('val', batch_size = args.batch_size, sample_rate= args.sample_rate, nmic = args.mic)
    #     # BJ: change number of workers to 8, according to Asteroid TAC setting
    #     tr_loader = AudioDataLoader(tr_dataset, batch_size=1, shuffle=args.shuffle, num_workers=0) #num_workers=0 for PC
    #     cv_loader = AudioDataLoader(cv_dataset, batch_size=1, num_workers=0) #num_workers=0 for PC
    if args.array_type == "fixed":
        tr_dataset = AudioDataset('tr', batch_size = args.batch_size, sample_rate= args.sample_rate, nmic = args.mic)
        cv_dataset = AudioDataset('val', batch_size = args.batch_size, sample_rate= args.sample_rate, nmic = args.mic)
        # BJ automatically split dataset with distributed sampler, so there is no need to shuffle dataset
        tr_loader = AudioDataLoader(tr_dataset, batch_size=1, shuffle=False, num_workers=0, sampler=DistributedSampler(tr_dataset)) #num_workers=0 for PC
        cv_loader = AudioDataLoader(cv_dataset, batch_size=1, shuffle=False, num_workers=0, sampler=DistributedSampler(cv_dataset)) #num_workers=0 for PC
    # elif args.array_type == "adhoc":
    #     tr_dataset = AdhocDataset('tr', batch_size = args.batch_size, sample_rate= args.sample_rate, max_mics = args.mic)
    #     cv_dataset = AdhocDataset('val', batch_size = args.batch_size, sample_rate= args.sample_rate, max_mics = args.mic)
    #     # BJ: change number of workers to 8, according to Asteroid TAC setting
    #     tr_loader = AdhocDataLoader(tr_dataset, batch_size=1, shuffle=args.shuffle, num_workers=0) #num_workers=0 for PC
    #     cv_loader = AdhocDataLoader(cv_dataset, batch_size=1, num_workers=0) #num_workers=0 for PC
    elif args.array_type == "adhoc":
        tr_dataset = AdhocDataset2('tr', sample_rate= args.sample_rate, max_mics = args.mic)
        cv_dataset = AdhocDataset2('val', sample_rate= args.sample_rate, max_mics = args.mic)
        # BJ: change number of workers to 8, according to Asteroid TAC setting
        tr_loader = AdhocDataLoader2(tr_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0) #num_workers=0 for PC
        cv_loader = AdhocDataLoader2(cv_dataset, batch_size=args.batch_size, num_workers=0) #num_workers=0 for PC

    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    # @BJ choose between fasnet/ifasnet model
    if args.model == "fasnet":
        model = FaSNet_TAC(enc_dim=args.enc_dim, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim, layer=args.layer, segment_size=args.segment_size, 
                            nspk=args.nspk, win_len=args.win_len, context_len=args.context_len, sr=args.sample_rate)
    elif args.model == "ifasnet":
        model = iFaSNet(enc_dim=args.enc_dim, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim, layer=args.layer, segment_size=args.segment_size, 
                           nspk=args.nspk, win_len=args.win_len, context_len=args.context_len, sr=args.sample_rate)
    elif args.model == "TF-GridNet":
        model = make_TF_GridNet(nmic=6, nspk=2, n_fft=256, D=32, B=6, I=8, J=1, H=256, E=4, L=4)
    
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of parameters:', k)
    
    #print(model)
    if args.use_cuda:
        # @BJ Constructing the DDP model
        # model = torch.nn.DataParallel(model)
        model.to(rank)
        model = DDP(model, device_ids=[rank])
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        
    
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args, rank)
    solver.train()

    destroy_process_group()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    # run on a single card
    # rank, world_size = 0, 1
    # train_worker(rank, world_size, args)

    # train on multiple cards
    world_size = torch.cuda.device_count()
    print(f"Number of GPUs detected: {world_size}")
    print("Spawning training processes ...")
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size)
