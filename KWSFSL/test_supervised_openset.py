import sys
import os
import json
from functools import partial
from tqdm import tqdm
import time 
import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
import torchvision
import torchnet as tnt

from utils import filter_opt
import models
from models.utils import get_model
import log as log_utils


def class2torchidx(labels):
    label_list = []
    for item in labels:
        label_index = ds.word_to_index[item]
        label_list.append([label_index])
    return torch.LongTensor(label_list)

def test_model(test_loader, model, unknow_id, force_ood_testdata=False):
    model.eval()
    data_loader = tqdm(test_loader, desc="Test")
    
    y_pred_tot = []
    y_true = []
    y_score = []
    y_pred_close_tot = []
    y_pred_ood_tot = []

    for sample in data_loader:
        x = sample['data']
        n_samples = x.size(0)
        labels = sample['label'] # labels
        if force_ood_testdata:
            labels = ['_unknown_' for item in labels]

        if isinstance(labels[0], str):
            labels = class2torchidx(labels).squeeze()

        if opt['data.cuda']:
            x = x.cuda()
            labels = labels.cuda()

        print(n_samples, labels, labels.size())
        _, output_i = model.loss_class(x,labels)
        print(output_i)

        p_y = output_i['p_y']

        # compute the probabilities
#            print('pred:',p_y, p_y.size() )
        _, y_pred = p_y.max(1)
        conf_val = p_y.gather(1, y_pred.unsqueeze(1)).squeeze().view(-1)

#        print(p_y.size())
        y_pred_ood = p_y[:,unknow_id]

        unknow_lab = torch.zeros(p_y.size(0)).add(unknow_id).long()
        y_pred_close = p_y[(1 - torch.nn.functional.one_hot(unknow_lab, p_y.shape[1])).bool()].reshape(p_y.shape[0], -1) 

        y_pred_ood_tot += y_pred_ood.tolist()
        y_pred_close_tot += y_pred_close.tolist()

#        print(p_y.size(), y_pred_close.size())

        y_pred_tot +=  y_pred.tolist()
        target_ids = labels.squeeze().tolist()
#        print(target_ids)
        y_true += [target_ids] if isinstance(target_ids, int) else target_ids

        conf_val = conf_val.tolist()
        y_score += [conf_val] if isinstance(conf_val, int) else conf_val

    return y_score, y_pred_tot, y_true, y_pred_close_tot, y_pred_ood_tot
        
        





if __name__ == '__main__':
    from argparser_kws import *
    opt = vars(parser.parse_args())

    # manual seed 
    torch.manual_seed(1234)
    if opt['data.cuda']:
        torch.cuda.manual_seed(1234)
    
    # load and prepare model to test
    if os.path.isfile(opt['model.model_path']):    
        model = torch.load(opt['model.model_path'])
        model.encoder.return_feat_maps = False # needed to ensure experiment backcompatibility

        loaded_opt_file = os.path.join(os.path.dirname(opt['model.model_path']), 'opt.json')
        with open(loaded_opt_file, 'r') as f:
            loaded_opt = json.load(f)

    else:
        raise ValueError("Model is not valid")
    print(model)

            
    if opt['data.cuda']:
        model.cuda()
                
    # import tasks: positive samples and optionative negative samples for open set
    # current limitations: tasks belongs to same dataset (separate eyword split)
    speech_args = filter_opt(opt, 'speech')
    dataset = opt['speech.dataset']
    data_dir = opt['speech.default_datadir'] 
    task = opt['speech.task'] 
    tasks = task.split(",")
    if len(tasks) == 2:
        pos_task, neg_task = tasks
    elif len(tasks) == 1:
        pos_task = tasks[0]
        neg_task = None

    if dataset == 'googlespeechcommand':
        from data.GSCSpeechData import GSCSpeechDataset
        ds = GSCSpeechDataset(data_dir, pos_task, opt['data.cuda'], speech_args)
        num_classes = ds.num_classes()
        opt['model.num_classes'] = num_classes
        print("The task {} of the {} Dataset has {} classes".format(
                pos_task, dataset, num_classes))
        
        ds_neg = None
        if neg_task is not None:
            ds_neg = GSCSpeechDataset(data_dir, neg_task, 
                    opt['data.cuda'], speech_args)
            print("The task {} is used for negative samples".format(
                    neg_task))       
    else:
        raise ValueError("Dataset not recognized")

    # Postprocess arguments FIXME
    opt['log.fields'] = opt['log.fields'].split(',')+['conf_corr','conf_wrong']
    opt['log.fields'] = ['aucROC','accuracy_pos', 'accuracy_neg', 'acc_prec95']

    # import stats
    meters = { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } 

    # get dataset labels 
    class_list = ds.words_list
    word_to_index = ds.word_to_index
    print(class_list, word_to_index)
    

    # get unknown index
    unk_idx = word_to_index['_unknown_'] if '_unknown_' in word_to_index.keys()  else None

    # test on positive dataset     
    print('\n Test on positive samples from classes: {}'.format(class_list))

    # load only samples from the target classes and not negative _unknown_
    query_loader = ds.get_iid_dataloader('testing', opt['fsl.test.batch_size'], 
        class_list = [x for x in class_list if 'unknown' not in x])
    y_score_pos, y_pred_pos, y_true_pos, y_pred_close_pos, y_pred_ood_pos = \
        test_model(query_loader, model, unk_idx)
    print(y_score_pos, y_pred_pos, y_true_pos, y_pred_close_pos, y_pred_ood_pos)

    # test on the negative dataset (_unknown_) if present    
    if ds_neg is not None:
        neg_loader = ds_neg.get_iid_dataloader('testing', opt['fsl.test.batch_size'])
        y_score_neg, y_pred_neg, y_true_neg, y_pred_close_neg, y_pred_ood_neg = \
            test_model(neg_loader, model, unk_idx, force_ood_testdata=True)
    else:
        y_score_neg, y_pred_neg, y_true_neg, y_pred_close_neg, y_pred_ood_neg = None, None, None, None, None

    # store and print metrics
    from test_fewshots_classifiers_openset import compute_metrics
    output_ep = compute_metrics(y_score_pos, y_pred_pos, y_true_pos, y_pred_close_pos, y_pred_ood_pos,
                    y_score_neg, y_pred_neg, y_true_neg, y_pred_close_neg, y_pred_ood_neg,
                    word_to_index, verbose=True)


    print(output_ep)

    output_log_file = 'eval_openset'
    output_file = os.path.join(os.path.dirname(opt['model.model_path']), output_log_file)
    print('Writing log to:', output_file)
    exit(0)
    with open(output_file, 'w') as fp:
        json.dump(output_ep, fp)


        
        

