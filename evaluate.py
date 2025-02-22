#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:09:09 2020

@author: yoonsanghyu
"""

# Created on 2018/12
# Author: Kaituo XU

import argparse
from mir_eval.separation import bss_eval_sources
from pit_criterion import cal_loss
from collections import OrderedDict
import numpy as np
import torch

from data import AdhocDataLoader, AdhocTestDataset, AudioDataset, EvalAudioDataLoader, OverlapRatioDataset, SpkAngleDataset
from FaSNet import FaSNet_TAC
from iFaSNet import iFaSNet


def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3: # [B, C, T]
            results.append(input[:,:length].view(C, -1).cpu().numpy())
        elif dim == 2:  # [B, T]  
            results.append(input[:length].view(-1).cpu().numpy())
    return results


parser = argparse.ArgumentParser('Evaluate separation performance using FaSNet + TAC')
parser.add_argument('--model_path', type=str, default='exp/tmp/temp_best.pth.tar', help='Path to model file created by training')
parser.add_argument('--cal_sdr', type=int, default=1, help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--use_cuda', type=int, default=1, help='Whether use GPU to separate speech')

# General config
# Task related
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')


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
# @BJ support result evaluation for different configuration
parser.add_argument('--split_type', default="overlap_ratio", type=str, choices=[None, 'overlap_ratio', 'speaker_angle'],
                    help = 'split a subset of test data to evaluate performance on specific kind of data,\
                            either can be "overlap_ratio" or "speaker_angle" or None')
parser.add_argument('--rg', default="0-25",type=str, choices=[None, '0-25', '25-50', '50-75', '75-100', '0-15', '15-45', '45-90', '90-180', 'all'],
                    help='which range of data do you want to evaluate')

# @BJ switch between adhoc and fixed array configuration
parser.add_argument('--array_type', default="adhoc", type=str, choices=['fixed', 'adhoc'],
                    help='Enable one of the training mode, either can be "fixed" or "adhoc"')

parser.add_argument('--num_mic', default=2, type=int, choices=[2,4,6],
                    help='number of micphones when evaluate in adhoc mode') # set it as 2/4/6 for different adhoc dataset evaluation

# @BJ change to iFasnet model
parser.add_argument('--model', default= "fasnet", type=str, choices=["ifasnet", "fasnet"],
                    help="choose different model for training")


def evaluate(args):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    # Load model

    if args.model == "fasnet":
        model = FaSNet_TAC(enc_dim=args.enc_dim, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim, layer=args.layer, segment_size=args.segment_size, 
                           nspk=args.nspk, win_len=args.win_len, context_len=args.context_len, sr=args.sample_rate)
    elif args.model == "ifasnet":
        model = iFaSNet(enc_dim=args.enc_dim, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim, layer=args.layer, segment_size=args.segment_size, 
                           nspk=args.nspk, win_len=args.win_len, context_len=args.context_len, sr=args.sample_rate)
    
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    
    # model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    model_info = torch.load(args.model_path)
    try:
        model.load_state_dict(model_info['model_state_dict'])
    except KeyError:
        state_dict = OrderedDict()
        for k, v in model_info['model_state_dict'].items():
            name = k.replace("module.", "")    # remove 'module.'
            state_dict[name] = v
        model.load_state_dict(state_dict)
    
    print(model)
    model.eval()

    # @BJ switch between adhoc mode and fixed mode
    if args.array_type == "adhoc":
        # usage example:"python evaluate.py --array_type adhoc --num_mic 6 --rg 75-100"
        dataset = AdhocTestDataset('test', batch_size=1, sample_rate=args.sample_rate, num_mic= args.num_mic, rg = args.rg)
        data_loader = AdhocDataLoader(dataset, batch_size=1, num_workers=8)
    elif args.array_type == "fixed":
        # @BJ Load data based on different split configuration
        if args.split_type == "overlap_ratio":
            dataset = OverlapRatioDataset('test', batch_size=1, sample_rate=args.sample_rate, nmic=args.mic, rg=args.rg)
        elif args.split_type == "speaker_angle":
            dataset = SpkAngleDataset('test', batch_size=1, sample_rate=args.sample_rate, nmic=args.mic, rg=args.rg)
        else:
            dataset = AudioDataset('test', batch_size = 1, sample_rate = args.sample_rate, nmic = args.mic)
        data_loader = EvalAudioDataLoader(dataset, batch_size=1, num_workers=8)
    
    sisnr_array=[]
    sdr_array=[]
    with torch.no_grad():
        for i, (data) in enumerate(data_loader):
            # Get batch data
            padded_mixture, mixture_lengths, padded_source = data

            # @BJ extract adhoc num_mic configuration
            # NOTE: if adhoc mode is enabled, num_mic will be appended after mixture_lengths to form a (2,B) shape tensor
            # if mixture_lengths has 2 dimentions, it simply means it's in adhoc mode
            if mixture_lengths.dim() == 2:
                num_mic = mixture_lengths[1]
                mixture_lengths = mixture_lengths[0]
            else:
                x = torch.rand(2, 4, 32000)
                num_mic = torch.zeros(1).type(x.type())

                # @BJ add support for multi-cards training
                num_mic = num_mic.repeat(len(range(torch.cuda.device_count())),1)
            
            if args.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
                
            # Forward
            estimate_source = model(padded_mixture, num_mic.long())  # [M, C, T]
            
            
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
                
            M,_,T = padded_mixture.shape    
            mixture_ref = torch.chunk(padded_mixture, args.mic, dim =1)[0] #[M, ch, T] -> [M, 1, T]
            mixture_ref = mixture_ref.view(M,T) #[M, 1, T] -> [M, T]
            
            mixture = remove_pad(mixture_ref, mixture_lengths)
            source = remove_pad(padded_source, mixture_lengths)
            estimate_source = remove_pad(reorder_estimate_source, mixture_lengths)
            
            # for each utterance
            for mix, src_ref, src_est in zip(mixture, source, estimate_source):
                print("Utt", total_cnt + 1)
                # Compute SDRi
                if args.cal_sdr:
                    avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                    total_SDRi += avg_SDRi
                    sdr_array.append(avg_SDRi)
                    print("\tSDRi={0:.2f}".format(avg_SDRi))
                # Compute SI-SNRi
                avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                total_SISNRi += avg_SISNRi
                sisnr_array.append(avg_SISNRi)
                total_cnt += 1
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
        
    np.save('sisnr.npy',np.array(sisnr_array))
    np.save('sdr.npy',np.array(sdr_array))
    print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    evaluate(args)