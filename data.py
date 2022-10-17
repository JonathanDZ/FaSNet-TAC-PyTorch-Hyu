# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:33:56 2020

@author: yoonsanghyu
"""


import os
import numpy as np
import librosa
import torch
import torch.utils.data as data

import json

# read 'tr' or 'val' or 'test' mixture path
def read_scp(opt_data, mix):

    mix_scp = 'data_script/{0}_{1}.scp'.format(opt_data, mix)
    lines = open(mix_scp, 'r').readlines()

    scp_dict= []
    for l in lines:
        # remove all white spaces before and after each line of strings and split it based on white spaces
        scp_parts = l.strip().split()
        scp_dict.append(scp_parts)
        
    return scp_dict

# put data path in batch
class AudioDataset(data.Dataset):
    def __init__(self, opt_data, batch_size = 3, sample_rate=16000, nmic=6):
        super(AudioDataset, self).__init__()
        '''
        opt_data : 'tr', 'val', 'test'
        batch_size : default 3
        sample_rate : 16000
        nmic : # of channel ex) fixed :6mic
        nsample : all sample/nmic
        
        '''
      
        #read data path
        mix_scp = read_scp(opt_data, 'mix')
        mix_path = mix_scp[0][0]
        nsample = int(mix_scp[1][0])

                
        
        minibatch = []
        mix = []
        end = 0
        while end < nsample:
            num_segments = 0  
            mix = []
            while num_segments < batch_size and end < nsample:
                end += 1
                mix.append(os.path.join(mix_path,'sample{0}'.format(end)))
                num_segments += 1
            minibatch.append([mix])
            
        self.minibatch = minibatch
                            
    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


# @BJ speaker angle dataset
class SpkAngleDataset(data.Dataset):
    def __init__(self, opt_data, batch_size = 3, sample_rate=16000, nmic=6, rg="0-15"):
        super(SpkAngleDataset, self).__init__()
        '''
        opt_data : 'tr', 'val', 'test'
        batch_size : default 3
        sample_rate : 16000
        nmic : # of channel ex) fixed :6mic
        nsample : all sample/nmic
        
        '''
      
        #read data path
        json_path = "/root/SpeechSeparation/TAC-FaSNet-DPRNN/data/pth_json"
        if rg == "0-15":
            json_file = os.path.join(json_path, "fixed-test-speaker-angle-0-15.json")
        elif rg == "15-45":
            json_file = os.path.join(json_path, "fixed-test-speaker-angle-15-45.json")
        elif rg == "45-90":
            json_file = os.path.join(json_path, "fixed-test-speaker-angle-45-90.json")
        elif rg == "90-180":
            json_file = os.path.join(json_path, "fixed-test-speaker-angle-90-180.json")
        
        with open(json_file, "r") as f:
            data_path = json.load(f)
        
        nsample = len(data_path)

        minibatch = []
        for i in range(nsample//batch_size):
            mix = []
            for j in range(batch_size):
                mix.append(data_path[batch_size*i+j])
            minibatch.append([mix])
        
        self.minibatch = minibatch
                            
    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)



# @BJ overlap ratio dataset
class OverlapRatioDataset(data.Dataset):
    def __init__(self, opt_data, batch_size = 3, sample_rate=16000, nmic=6, rg="0-25"):
        super(OverlapRatioDataset, self).__init__()
        '''
        opt_data : 'tr', 'val', 'test'
        batch_size : default 3
        sample_rate : 16000
        nmic : # of channel ex) fixed :6mic
        nsample : all sample/nmic
        
        '''
      
        #read data path
        json_path = "/root/SpeechSeparation/TAC-FaSNet-DPRNN/data/pth_json"
        if rg == "0-25":
            json_file = os.path.join(json_path, "fixed-test-overlap-ratio-0-25.json")
        elif rg == "25-50":
            json_file = os.path.join(json_path, "fixed-test-overlap-ratio-25-50.json")
        elif rg == "50-75":
            json_file = os.path.join(json_path, "fixed-test-overlap-ratio-50-75.json")
        elif rg == "75-100":
            json_file = os.path.join(json_path, "fixed-test-overlap-ratio-75-100.json")
        
        with open(json_file, "r") as f:
            data_path = json.load(f)
        
        nsample = len(data_path)

        minibatch = []
        for i in range(nsample//batch_size):
            mix = []
            for j in range(batch_size):
                mix.append(data_path[batch_size*i+j])
            minibatch.append([mix])
        
        self.minibatch = minibatch
                            
    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


# @BJ adhoc train/validation dataset
class AdhocDataset(data.Dataset):
    def __init__(self, opt_data, batch_size = 3, sample_rate=16000, max_mics=6):
        super(AdhocDataset, self).__init__()
        '''
        opt_data : 'tr', 'val', 'test'
        batch_size : default 3
        sample_rate : 16000
        max_mic : # of max channel ex) adhoc :nmic <= 6mic
        nsample : all sample/nmic
        
        '''

        # read data path
        json_path = "/root/SpeechSeparation/TAC-FaSNet-DPRNN/data/pth_json"

        if opt_data == 'tr':
            json_file = os.path.join(json_path, "adhoc-train-all.json")
        elif opt_data == 'val':
            json_file = os.path.join(json_path, "adhoc-validation-all.json")
        
        with open(json_file, "r") as f:
            data_path = json.load(f)
        
        nsample = len(data_path)

        minibatch = []
        for i in range(nsample//batch_size):
            mix = []
            for j in range(batch_size):
                mix.append(data_path[batch_size*i+j])
            minibatch.append([mix])
        
        # minibatch[index] [[[path, nmic],[path, nmic],[path, nmic]]]
        # (1,1,3,2)
        self.minibatch = minibatch

                            
    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


    
# read wav file in batch for tr, val      
class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
        
def _collate_fn(batch):
    
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mix_torch: B x ch x T, torch.Tensor
        ilens_torch : B, torch.Tentor
        src_torch: B x C x T, torch.Tensor
        
    ex)
    torch.Size([3, 6, 64000])
    tensor([64000, 64000, 64000], dtype=torch.int32)
    torch.Size([3, 2, 64000])
    """
    
    sr = 16000
    nmic =6

    # batch (1,1,3)
    assert len(batch) == 1
    
    total_mix = []
    total_src = []
    # i is the path to /sample* directory
    for i in batch[0][0]:
        
        mix_list=[]
        # # TODO: I can change here to support adhoc configuration
        for n in range(nmic):
            mix_path = os.path.join(i,'mixture_mic{0}.wav'.format(n+1)) 

            mix, _ = librosa.load(mix_path, sr)
            mix_list.append(mix)
                      
            # we only use first mic fro source signal            
            if n ==0:
                s1_path = os.path.join(i,'spk1_mic{0}.wav'.format(n+1))
                s2_path = os.path.join(i,'spk2_mic{0}.wav'.format(n+1))
                
                s1, _ = librosa.load(s1_path, sr)
                s2, _ = librosa.load(s2_path, sr)
                
                src_list = [s1,s2]


        src_np = np.asarray(src_list,dtype=np.float32)
        mix_np = np.asarray(mix_list,dtype=np.float32)
        
    
        total_mix.append(mix_np)
        total_src.append(src_np)

        
    total_mix_np = np.asarray(total_mix,dtype=np.float32)
    total_src_np= np.asarray(total_src,dtype=np.float32)

    mix_torch = torch.from_numpy(total_mix_np)
    src_torch = torch.from_numpy(total_src_np)


    ilens = np.array([mix.shape[1] for mix in mix_torch])
    ilens_torch = torch.from_numpy(ilens)

    return mix_torch, ilens_torch, src_torch


# read wav file in batch for test   
class EvalAudioDataLoader(data.DataLoader):
    
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalAudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval
        
def _collate_fn_eval(batch):

    # batch should be located in list    
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mix_torch: B x ch x T, torch.Tensor
        ilens_torch : B, torch.Tentor
        src_torch: B x C x T, torch.Tensor
        
    ex)
    torch.Size([3, 6, 64000])
    tensor([64000, 64000, 64000], dtype=torch.int32)
    torch.Size([3, 2, 64000])
    """
    
    sr = 16000
    nmic =6
    
    total_mix = []
    total_src = []
    for i in batch[0][0]:
        
        mix_list=[]        
        for n in range(nmic):
            mix_path = os.path.join(i,'mixture_mic{0}.wav'.format(n+1)) 

            mix, _ = librosa.load(mix_path, sr)
            mix_list.append(mix)
                      
            # we only use first mic fro source signal            
            if n ==0:
                s1_path = os.path.join(i,'spk1_mic{0}.wav'.format(n+1))
                s2_path = os.path.join(i,'spk2_mic{0}.wav'.format(n+1))
                
                s1, _ = librosa.load(s1_path, sr)
                s2, _ = librosa.load(s2_path, sr)
                
                src_list = [s1,s2]


        src_np = np.array(src_list)
        mix_np = np.array(mix_list)
        
    
        total_mix.append(mix_np)
        total_src.append(src_np)

        
    total_mix_np = np.array(total_mix)
    total_src_np= np.array(total_src)

    mix_torch = torch.from_numpy(total_mix_np)
    src_torch = torch.from_numpy(total_src_np)


    ilens = np.array([mix.shape[1] for mix in mix_torch])
    ilens_torch = torch.from_numpy(ilens)

    return mix_torch, ilens_torch, src_torch



# @BJ read adhoc wav file in batch for tr, val
# NOTE: for adhoc training mode, it returns a stack of ilens & num_mic instead of only ilens info
class AdhocDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AdhocDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_adhoc
        
def _collate_fn_adhoc(batch):
    
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mix_torch: B x ch x T, torch.Tensor
        [ilens_torch, num_mic] : 2 x B, torch.Tentor
        src_torch: B x C x T, torch.Tensor
        
    ex)
    torch.Size([3, 6, 64000])
    tensor([[64000, 64000, 64000], [6, 6, 6]], dtype=torch.int32)
    torch.Size([3, 2, 64000])
    """
    
    sr = 16000
    max_mics = 6

    # batch (1,1,3,2)
    assert len(batch) == 1
    
    total_mix = []
    total_src = []

    num_mic = []
    # i is the path to /sample* directory
    for i, nmic in batch[0][0]:
        
        mix_list = []
        # TODO: I can change here to support adhoc configuration
        for n in range(nmic):
            mix_path = os.path.join(i,'mixture_mic{0}.wav'.format(n+1)) 

            mix, _ = librosa.load(mix_path, sr)
            mix_list.append(mix)
                      
            # we only use first mic for source signal            
            if n == 0:
                s1_path = os.path.join(i,'spk1_mic{0}.wav'.format(n+1))
                s2_path = os.path.join(i,'spk2_mic{0}.wav'.format(n+1))
                
                s1, _ = librosa.load(s1_path, sr)
                s2, _ = librosa.load(s2_path, sr)
                
                src_list = [s1,s2]


        src_np = np.asarray(src_list,dtype=np.float32)
        mix_np = np.asarray(mix_list,dtype=np.float32)

        # padded to 'max_mics'
        if mix_np.shape[0] < max_mics:
            dummy = np.zeros((max_mics - mix_np.shape[0], mix_np.shape[-1]))
            mix_np = np.concatenate((mix_np, dummy), axis=0)
        
    
        total_mix.append(mix_np)
        total_src.append(src_np)

        num_mic.append(nmic)

        
    total_mix_np = np.asarray(total_mix,dtype=np.float32)
    total_src_np= np.asarray(total_src,dtype=np.float32)
    num_mic = np.asarray(num_mic, dtype=np.int64)

    mix_torch = torch.from_numpy(total_mix_np)
    src_torch = torch.from_numpy(total_src_np)
    num_mic = torch.from_numpy(num_mic)


    ilens = np.array([mix.shape[1] for mix in mix_torch])
    ilens_torch = torch.from_numpy(ilens)

    # stack ilens and num_mics
    config_info = torch.stack((ilens_torch, num_mic))

    return mix_torch, config_info, src_torch