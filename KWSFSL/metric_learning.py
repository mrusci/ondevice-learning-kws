import sys
import os
import json
from functools import partial
from tqdm import tqdm
import time 
import numpy as np
from shutil import copyfile


# import torch 
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
import torchnet as tnt

# import subpackages
from utils import filter_opt
import models
from models.utils import get_model
import log as log_utils


# needed by the computing infrastructure, you can remove it!
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('_CONDOR_AssignedGPUs', 'CUDA0').replace('CUDA', '')


if __name__ == '__main__':
    
    # read and post-process options
    from parser_kws import *
    args = parser.parse_args()

    opt = vars(parser.parse_args())
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = ['loss']
    speech_args = filter_opt(opt, 'speech')
    model_opt = filter_opt(opt, 'model')
    model_type = model_opt['model_name']


    ##################################################
    #    Prepare and Load the model
    ##################################################
    
    print("Load Non Initilaized Model")
    
    # prepare preprocessing
    if opt['model.preprocessing'] == 'mfcc':
        print('Setup Preprocessing configuration structure')
        model_opt['mfcc'] = { 
            'window_size_ms': speech_args['window_size'],
            'window_stride_ms': speech_args['window_stride'],
            'sample_rate': speech_args['sample_rate'],
            'n_mfcc': speech_args['n_mfcc'],
            'feature_bin_count': speech_args['num_features']
        }
        
    # Metric Learning Parameters
    n_way = opt['train.n_way']
    n_support = opt['train.n_support']
    n_query = opt['train.n_query']
    n_episodes = opt['train.n_episodes']

    # preparare loss
    print('Loss function: ', opt['train.loss'])
    model_opt['loss'] = {'type': opt['train.loss'], 'margin':  opt['train.margin']}
    if opt['train.loss'] == 'prototypical' or opt['train.loss'] == 'angproto':
        model_opt['loss']['n_support'] = n_support
        model_opt['loss']['n_query'] = n_query
    elif opt['train.loss'] == 'peeler' or opt['train.loss'] == 'dproto':
        model_opt['loss']['n_support'] = opt['train.n_support']
        model_opt['loss']['n_query'] = opt['train.n_query']
        model_opt['loss']['n_way_u'] = opt['train.n_way_u']



    #load the model
    model = get_model(model_opt)
    print(model)

    # initialize weights from a pretrained model store in model.model_path (not used currently)
    if os.path.isfile(opt['model.model_path']):   
        print('Load Pretrained Model from', model.model_path)
        enc_model = torch.load(opt['model.model_path'])
        model.encoder.load_state_dict(enc_model.encoder.state_dict())
                
    # move to cuda
    if opt['data.cuda']:
        model.cuda()     
        if  'mfcc' in model_opt.keys():
            model.preprocessing.mfcc.cuda()

    ##################################################
    #    Prepare and Load the training dataset
    ##################################################
    # import training and validation tasks
    # validation is optional and it is expected to be from the same dataset
    dataset = opt['speech.dataset']
    data_dir = opt['speech.default_datadir'] 
    train_task = opt['speech.task'] 
    print('Train Dataset: ', train_task)
    
    #prepare datasets (supported:  'googlespeechcommand' , 'MSWC')
    if dataset == 'googlespeechcommand':
        from data.GSCSpeechData import GSCSpeechDataset
        ds_tr = GSCSpeechDataset(data_dir, train_task, opt['data.cuda'], speech_args)
    elif dataset == 'MSWC':
        from data.MSWCData import MSWCDataset
        ds_tr = MSWCDataset(data_dir, train_task, False, speech_args)
    else:
        raise ValueError("Dataset not recognized")

    #number of classes of the training task
    num_classes_tr = ds_tr.num_classes()
    print("The training task {} of the {} Dataset has {} classes".format(dataset, train_task, num_classes_tr))
    n_way_tr = min(max(n_way, 0), num_classes_tr) # clamp n_way based on availbale classes

    ##################################################
    #    Prepare training options
    ##################################################
    
    # import stats
    meters = { 'train': { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } }

    # setup the optimizer
    optim_method = getattr(optim, opt['train.optim_method'])
    optim_config = { 'lr': opt['train.learning_rate'],
                     'weight_decay': opt['train.weight_decay'] }
    optimizer = optim_method(model.parameters(), **optim_config)
    
    scheduler = lr_scheduler.StepLR(optimizer, opt['train.decay_every'], gamma=0.5)

    # setup experiment directory or load checkpoint, if any
    opt['log.exp_dir'] = os.path.join('./results', opt['log.exp_dir'])
    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')
    checkpoint_file = os.path.join(opt['log.exp_dir'], 'checkpoint.pt')
    
    ##################################################
    #    Load checkpoint (if any)
    ##################################################
    if os.path.isfile(checkpoint_file):
        print('Found Checkpoint!')
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['start_epoch']
        best_loss = checkpoint['best_loss']
        wait = checkpoint['wait']
        start_episode = checkpoint['start_episode']
    else:
        if not os.path.isdir(opt['log.exp_dir']):
            os.makedirs(opt['log.exp_dir'])
        #trace file
        if os.path.isfile(trace_file):
            os.remove(trace_file)
            
        # save opts
        with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
            json.dump(opt, f)
            f.write('\n')
        
        start_epoch = 0
        best_loss = np.inf
        wait = 0    
        start_episode = 0
    
    
    ##################################################
    #    Launch training
    ##################################################
    max_epoch = opt['train.epochs']
    print("Training model {} in a few-shot setting ({}-way | {}-shots) for {} episodes and {}\
            epochs on the task {} of the Dataset {}".format(model_opt['encoding'], n_way,
            n_support, n_episodes, max_epoch, train_task, dataset))
    
    cuda = opt['data.cuda']
    train_patience = opt['train.patience']
    stop = False
    epoch = start_epoch

    model.train()

    while epoch < max_epoch and not stop:
        # get episode loaders 
        episodic_loader = ds_tr.get_episodic_dataloader('training', n_way_tr, 
            n_support+n_query, n_episodes-start_episode)

        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        epoch_size = len(episodic_loader)

        ep_idx = start_episode
        for samples in tqdm(episodic_loader,desc="Epoch {:d} train".format(epoch + 1)):
            samples_ep = samples['data']
            if cuda:
                samples_ep = samples_ep.cuda()
            optimizer.zero_grad()
            loss, output = model.loss(samples_ep)
            loss.backward()

            optimizer.step()

            for field, meter in meters['train'].items():
                meter.add(output[field])
            
            ep_idx+=1

            # save checkpoint every 10 episodes
            # if cuda is used, the checkpoint reload cuda tensors
            stored_ckpt = False
            if (ep_idx)%10 == 0:
                # to avoid saving issues, try first to save into a tmp file. if success copy
                # (this may be avoided. I did this for some issues with the nfs!!)
                checkpoint_file_tmp = os.path.join(opt['log.exp_dir'], 'checkpoint_tmp.pt')
                while stored_ckpt is False:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'start_epoch': epoch,
                        'start_episode': ep_idx, 
                        'best_loss': best_loss,
                        'wait': wait,
                        }, checkpoint_file_tmp)

                    # check if it is correctly stored
                    try:
                        torch.load(checkpoint_file_tmp)
                    except EOFError:
                        print('Error Storing Ckpt at episode {} of epoch {}'.format(
                                ep_idx, epoch))                    
                    else:
                        copyfile(checkpoint_file_tmp, checkpoint_file)
                        stored_ckpt = True

        # end epoch
        start_episode = 0
        scheduler.step()

        # log at the end of the epoch
        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:02d}: {:s}".format(epoch, log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = epoch
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        model.cpu()
        torch.save(model, os.path.join(opt['log.exp_dir'], 'best_model.pt'))
        if cuda:
            model.cuda()
        
        epoch += 1


