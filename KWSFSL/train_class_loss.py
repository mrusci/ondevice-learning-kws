import sys
import os
import json
from functools import partial
from tqdm import tqdm
import time
import datetime 
import numpy as np

# needed by the computing infrastructure, you can remove it!
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('_CONDOR_AssignedGPUs', 'CUDA0').replace('CUDA', '')

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
import torchvision
import torchnet as tnt

from utils import filter_opt
import models
from models.utils import get_model
import log as log_utils


def train(
        model,
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler,     
        training_options, 
        meters, 
        cuda, 
        exp_dir, 
        trace_file,
        checkpoint_file 
    ):
    
    epoch = training_options['epoch']
    max_epoch = training_options['max_epoch']
    train_patience = training_options['train_patience']
    best_loss = training_options['best_loss']
    wait = training_options['wait']
        
    # start the training loop from start epoch
    stop = False    
    while epoch < max_epoch and not stop:
        model.train()
        
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        
        epoch_size = len(train_loader)
        
        for sample in tqdm(train_loader, desc="Epoch {:d} train".format(epoch + 1)):
            x = sample['data'] # input features
            labels = sample['label_idx'] # label
            if cuda:
                x = x.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            loss, output = model.loss_class(x, labels)
            loss.backward()

            optimizer.step()
            
            for field, meter in meters['train'].items():
                meter.add(output[field])

        # end epoch
        scheduler.step()
        epoch += 1

        if val_loader is not None:
            evaluate(model, val_loader, meters['val'], cuda,
                     desc="Epoch {:d} valid".format(epoch))

        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:02d}: {:s}".format(epoch, log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = epoch
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        if val_loader is not None:
            if meter_vals['val']['loss'] < best_loss:
                best_loss = meter_vals['val']['loss']
                print("==> best model (loss = {:0.6f}), saving model...".format(best_loss))

                model.cpu()
                torch.save(model, os.path.join(exp_dir, 'best_model.pt'))
                if cuda:
                    model.cuda()

                wait = 0
            else:
                wait += 1

                if wait > train_patience:
                    print("==> patience {:d} exceeded".format(train_patience))
                    stop = True
        else:
            model.cpu()
            torch.save(model, os.path.join(exp_dir, 'best_model.pt'))
            if cuda:
                model.cuda()
        
        # save checkpoint
        # if cuda is used, the checkpoint reload cuda tensors
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'start_epoch': epoch,
            'best_loss': best_loss,
            'wait': wait,
            }, checkpoint_file)

        
        

def evaluate(model, data_loader, meters, cuda, desc=None):
    model.eval()

    for field,meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:

        x = sample['data'] # input features
        labels = sample['label_idx'] # label
        if cuda:
            x = x.cuda()
            labels = labels.cuda()

        _, output = model.loss_class(x, labels)
        for field, meter in meters.items():
            meter.add(output[field])

    return meters



if __name__ == '__main__':
    from argparser_kws import *
    args = parser.parse_args()

    opt = vars(parser.parse_args())

    # manual seed 
#    torch.manual_seed(1234)
#    if opt['data.cuda']:
#        torch.cuda.manual_seed(1234)
        
    # Postprocess arguments FIXME
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')
    
        
    # import task
    speech_args = filter_opt(opt, 'speech')
    dataset = opt['speech.dataset']
    data_dir = opt['speech.default_datadir'] 
    task = opt['speech.task'] 
    if dataset == 'googlespeechcommand':
        from data.GSCSpeechData import GSCSpeechDataset
        ds = GSCSpeechDataset(data_dir, task, opt['data.cuda'], speech_args)
    elif dataset == 'MSWC':
        from data.MSWCData import MSWCDataset
        ds = MSWCDataset(data_dir, task, False, speech_args)
    else:
        raise ValueError("Dataset not recognized")
        
    num_classes = ds.num_classes()
    print("The task {} of the {} Dataset has {} classes".format(task, dataset, num_classes))
        
    # import dataloaders
    train_loader = ds.get_iid_dataloader('training', opt['train.batch_size'])
    val_loader = ds.get_iid_dataloader('validation', opt['train.batch_size'])
    
    #import model
    model_opt = filter_opt(opt, 'model')
    if model_opt['model_name'] == 'e2e_conv':
        # setup n_classes od the classifier
        model_opt['num_classes'] = num_classes

    elif model_opt['model_name'] == 'repr_conv':
        # setup loss
        model_opt['loss'] = {
                'type': opt['train.loss'], 
                'margin':  opt['train.margin'],
                'n_classes':  num_classes,}
    else:
        raise ValueError("Not valid Model Type")     

    # prepare preprocessing
    if model_opt['preprocessing'] == 'mfcc':
        print('MFCC preprocessing')
        model_opt['mfcc'] = { 
            'window_size_ms': speech_args['window_size'],
            'window_stride_ms': speech_args['window_stride'],
            'sample_rate': speech_args['sample_rate'],
            'n_mfcc': speech_args['n_mfcc'],
            'feature_bin_count': speech_args['num_features']
        }

    # setup loss 
    model = get_model(model_opt)

    
    print(model)

    #move to cuda
    cuda = opt['data.cuda']
    if cuda:
        model.cuda()
        if  'mfcc' in model_opt.keys():
            model.preprocessing.mfcc.cuda()


    # import stats
    meters = { 'train': { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } }
    if val_loader is not None:
        meters['val'] = { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] }
        
    # setup the optimizer
    optim_method = getattr(optim, opt['train.optim_method'])
    optim_config = { 'lr': opt['train.learning_rate'],
                     'weight_decay': opt['train.weight_decay'] }


    # setup optimizer and schedule
    optimizer = optim_method(model.parameters(), **optim_config)
    scheduler = lr_scheduler.StepLR(optimizer, opt['train.decay_every'], gamma=0.5)
    
    
    # setup experiment directory
    opt['log.exp_dir'] = os.path.join('./results', opt['log.exp_dir'])
    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')
    checkpoint_file = os.path.join(opt['log.exp_dir'], 'checkpoint.pt')
    
    if os.path.isfile(checkpoint_file):
        print('Found Checkpoint!')
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['start_epoch']
        best_loss = checkpoint['best_loss']
        wait = checkpoint['wait']

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


        
    training_options= { 'epoch': start_epoch,
                       'max_epoch': opt['train.epochs'],
                       'train_patience': opt['train.patience'],
                       'best_loss': best_loss,
                       'wait': wait
                     }

    
    # launch teh training
    start = time.time()
    train(
      model, 
      train_loader, 
      val_loader, 
      optimizer, 
      scheduler,     
      training_options, 
      meters, 
      cuda, 
      opt['log.exp_dir'], 
      trace_file,
      checkpoint_file 
    )
    end = time.time()
    elapsed = str(datetime.timedelta(seconds= end-start))
    print("Total Time: {}".format(elapsed))
    