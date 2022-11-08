# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:20:37 2020

Created on 2018/12
Author: Kaituo XU

Edited by: yoonsanghyu  2020/04


"""

import os
import time

import torch
import numpy as np

from pit_criterion import cal_loss

import json
from utility.sdr import batch_SDR_torch

class Solver(object):
    
    def __init__(self, data, model, optimizer, args, rank):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer
        self.rank = rank
        
        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm

        # @BJ lr decay
        self.lr_decay = False

        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        # logging
        self.print_freq = args.print_freq
        # loss
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            cont = torch.load(self.continue_from)
            self.start_epoch = cont['epoch']
            # @BJ access model's parameter with model.module
            self.model.module.load_state_dict(cont['model_state_dict'])
            self.optimizer.load_state_dict(cont['optimizer_state'])
            torch.set_rng_state(cont['trandom_state'])
            np.random.set_state(cont['nrandom_state'])
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            optim_state = self.optimizer.state_dict()
            print('epoch start Learning rate: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
            print("Training...")
            
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0:5d} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            # @BJ only save the model on GPU 0 with DDP training
            if self.rank == 0 and self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save({
                    'epoch': epoch+1,
                    # @BJ access model's parameter with model.module
                    'model_state_dict': self.model.module.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, file_path)
                print('Saving checkpoint model to %s' % file_path)

            
            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Adjust learning rate (halving)
            if self.half_lr:
                if val_loss >= self.best_val_loss:
                    self.val_no_impv += 1
                    # @BJ comment out following 2 lines to stop halving
                    if self.val_no_impv >= 3:
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No imporvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            
            # @BJ the learning rate is decayed by 0.98 for every 2 epochs
            if self.lr_decay:
                if (epoch - 1) % 2 == 0:
                    optim_state = self.optimizer.state_dict()
                    optim_state['param_groups'][0]['lr'] = \
                        optim_state['param_groups'][0]['lr'] * 0.98
                    self.optimizer.load_state_dict(optim_state)
                    print('Learning rate adjusted to: {lr:.6f}'.format(
                        lr=optim_state['param_groups'][0]['lr']))

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_file_path = os.path.join(
                    self.save_folder, 'temp_best.pth.tar')
                torch.save({
                    'epoch': epoch+1,
                    # @BJ access model's parameter with model.module
                    'model_state_dict': self.model.module.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, best_file_path)
                print("Find better validated model, saving to %s" % best_file_path)


    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0
        # total_SDR_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        # @BJ Calling the set_epoch() method on the DistributedSampler at the beginning of each epoch 
        #     is necessary to make shuffling work properly across multiple epochs
        data_loader.sampler.set_epoch(epoch)

        for i, (data) in enumerate(data_loader):
            
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
            
            if self.use_cuda:
                # @BJ move all input tensor to GPU
                padded_mixture = padded_mixture.to(self.rank)
                mixture_lengths = mixture_lengths.to(self.rank)
                padded_source = padded_source.to(self.rank)
                num_mic = num_mic.to(self.rank)
                self.model.to(self.rank)
                window = torch.hann_window(window_length=256).to(self.rank)

            

            # estimate_source = self.model(padded_mixture, num_mic.long())
            # @BJ Manually pass "window" argument to fix a torch.stft bug
            # see: https://github.com/pytorch/pytorch/issues/30865
            estimate_source = self.model(padded_mixture, window)

            # max_SDR = batch_SDR_torch(estimate_source, padded_source)
            # SDR_loss = 0 - torch.mean(max_SDR)
                        
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)


            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                #optim_state = self.optimizer.state_dict()
                #print('Learning rate adjusted to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
                print('Epoch {0:3d} | Iter {1:5d} | Average Loss {2:3.3f} | '
                      'Current Loss {3:3.6f} | {4:5.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)
        
            
        del padded_mixture, mixture_lengths, padded_source, num_mic,\
            loss, max_snr, estimate_source, reorder_estimate_source
            
        if self.use_cuda: torch.cuda.empty_cache()

        return total_loss / (i + 1)
