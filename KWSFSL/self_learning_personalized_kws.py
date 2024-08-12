# Copyright © 2024 Manuele Rusci

# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the “Software”), 
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.


# imports
import argparse
import json
import os
from pathlib import Path
import copy
import random
from itertools import combinations


# needed for our comptue infrstructure. has not impact on other systems. 
# need to be call before import pytorch
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('_CONDOR_AssignedGPUs', 'CUDA0').replace('CUDA', '')


# import pytorch modules
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from classifiers.NCM import NearestClassMean # classifiers
import torch.optim as optim

##################################################
####### dataset class 
##################################################
class WakeWordDataset:
    def __init__(self, data_dir_pos, data_dir_neg, args, dataset):

        self.sr = args['sample_rate']
        self.out_dir = args['out_dir']
        self.data_dir_pos = data_dir_pos
        self.data_dir_neg = data_dir_neg
        self.dataset = dataset

        if self.dataset == 'heysnips':
            # load original train/val/test partition of hey snips 
            train_meta = json.load(open(os.path.join(self.data_dir_pos, "train.json"), 'r'))
            dev_meta = json.load(open(os.path.join(self.data_dir_pos, "dev.json"), 'r'))
            test_meta = json.load(open(os.path.join(self.data_dir_pos, "test.json"), 'r'))
            self.test_meta = test_meta  + dev_meta
            self.train_meta = train_meta

        elif self.dataset == 'heysnapdragon':

            # load hey snips for the negatives
            train_meta = json.load(open(os.path.join(self.data_dir_neg, "train.json"), 'r'))
            dev_meta = json.load(open(os.path.join(self.data_dir_neg, "dev.json"), 'r'))
            test_meta = json.load(open(os.path.join(self.data_dir_neg, "test.json"), 'r'))
            for item in train_meta + dev_meta + test_meta:
                item['audio_file_path'] = os.path.join(self.data_dir_neg, item['audio_file_path'])

            # load positive data and configuration 
            path_dir = self.data_dir_pos
            spk_list = os.listdir(path_dir)
            spks_test =  20 # used for test
            spks_train = int(len(spk_list)) - spks_test # rest of the speakers are used for (unsupervised) training
            
            # train set (first spks_train speakers + negative data from train_meta of heysnips)
            list_data = []
            for item in spk_list[:spks_train]:
                spk_dir = os.path.join(path_dir, item)
                for file in os.listdir(spk_dir):
                    path_item = os.path.join(path_dir, item, file)
                    file_struct = {'duration': 1,
                        'worker_id': item,
                        'audio_file_path': path_item,
                        'id': os.path.splitext(os.path.basename(file))[0],
                        'is_hotword': 1
                    }
                    list_data.append(file_struct)
            for item in train_meta:
                if item['is_hotword'] != 1:
                    list_data.append(item)
            random.shuffle(list_data)
            self.train_meta = list_data

            # test set
            list_data = []
            for item in spk_list[spks_train:]:
                spk_dir = os.path.join(path_dir, item)
                for file in os.listdir(spk_dir):
                    path_item = os.path.join(path_dir, item, file)
                    file_struct = {'duration': 1,
                        'worker_id': item,
                        'audio_file_path': path_item,
                        'id': os.path.splitext(os.path.basename(file))[0],
                        'is_hotword': 1
                    }
                    list_data.append(file_struct)

            for item in test_meta + dev_meta:
                if item['is_hotword'] != 1:
                    list_data.append(item)
            random.shuffle(list_data)
            self.test_meta = list_data
        else:
            raise ValueError("Dataset {} not supported!".format(dataset))
    


    def get_person_splits(self, min_pos_sample=10, n_train_pos_samples = 3, \
                          n_train_neg_samples = 10, min_sample_duration_sec = 1):
        ## organize *test* dataset w.r.t to speaker IDs

        #params
        self.min_pos_sample = min_pos_sample    # retain speaker data if #samples > min_pos_sample
        self.n_train_pos_samples = n_train_pos_samples # pos samples used for training (e.g. few-shot classifier initialization)
        self.n_train_neg_samples = n_train_neg_samples # neg samples used for training (e.g. few-shot classifier initialization)
        
        user_data = {} # dictionary of speaker data
        neg_data = [] #list of neg data 
        self.tot_time_neg = 0
        self.cnt_spk = 0
        self.n_pos_samples = 0
        self.n_neg_samples = 0

        # final lists of negative and per-speaker positive data, with train (i.e. few-shot initialization) and test split 
        self.pos_data_eval = []
        self.neg_data_eval = []

        # order the test dataset by speaker
        for item in self.test_meta:
            if item['duration'] < min_sample_duration_sec:  # discard short sample
                continue
            if item['is_hotword'] == 1:
                user_id = item['worker_id']
                if user_id in user_data:
                    user_data[user_id].append(item)
                else:
                    user_data[user_id] = [item]
            else:
                neg_data.append(item)
                self.tot_time_neg += item['duration']

        # get the positive splits
        for key, item in user_data.items():
            random.shuffle(item)
            if len(item) > self.min_pos_sample:
                self.cnt_spk += 1
                self.n_pos_samples+=len(item)
                self.pos_data_eval.append({'train': item[:self.n_train_pos_samples], 'test': item[self.n_train_pos_samples:]})

        # get the negative split
        self.n_neg_samples = len(neg_data)
        random.shuffle(neg_data)
        self.neg_data_eval = {'train': neg_data[:self.n_train_neg_samples], 'test': neg_data[self.n_train_neg_samples:]}


    def get_train_splits(self, adapt_set_ratio, num_continual_set=1):
        self.num_continual_set = num_continual_set
        tr_len = len(self.train_meta)
        adapt_set_item = int(tr_len*adapt_set_ratio)
        notused_set_item = int(tr_len-(num_continual_set*adapt_set_item))
        random_ass = [-1 for _ in range(notused_set_item)]
        for i in range(num_continual_set):
            random_ass += [int(i+1) for _ in range(adapt_set_item)]
        random.shuffle(random_ass)

        print('[Global Train data] Producing {} set of {} samples'.format(num_continual_set,adapt_set_item))
        continual_meta = [[] for x in range(num_continual_set)] 
        for i,item in enumerate(random_ass):
            if item > 0:
                continual_meta[item-1].append(self.train_meta[i])
        return continual_meta

    def print_stats(self, dataset):
        cnt_kw = 0
        cnt_nkw = 0
        sec_kw = 0
        sec_nkw = 0
        for item in dataset:
            samples, _ = librosa.load( item['audio_file_path'] , sr=self.sr)
            msec_samples = samples.shape[0] / self.sr
            if item['is_hotword'] == 1:
                cnt_kw += 1
                sec_kw += msec_samples
            else:
                cnt_nkw += 1
                sec_nkw += msec_samples
        print('Content: {} kwd ({} min) and {} no kwd ({} min)'.format(cnt_kw, int(sec_kw/60), cnt_nkw, int(sec_nkw/60))) 
        return cnt_kw, cnt_nkw


##################################################
####### test functions
##################################################

def compute_power(signal):
    power = np.sum( np.power(signal,2) )
    return power


def find_time_of_power(speech, perc=0.9, perc_blocks=0.05):
    len_speech = len(speech)
    len_blk = int(len_speech * perc_blocks)
    power = compute_power(speech)

    ptr_0 = 0
    ptr_1 = len_speech

    new_ptr_0 = ptr_0
    new_ptr_1 = ptr_1
    while True:

        if compute_power(speech[ptr_0:ptr_0+len_blk]) > compute_power(speech[ptr_1-len_blk:ptr_1]):
            new_ptr_1 = ptr_1 - len_blk
        else:
            new_ptr_0 = ptr_0+len_blk

        if compute_power(speech[new_ptr_0:new_ptr_1]) < 0.9 * power:
            return (ptr_1 - ptr_0)
        else:
            ptr_0 = new_ptr_0
            ptr_1 = new_ptr_1
        
        assert ptr_1 > ptr_0 , "Value Error"


# temporal filter of dist(t)
# a_f is the filter lenght
def filter_signal(dist_array, a_f=1.0, sub_sample=1):
    kw_dist = copy.deepcopy(dist_array)
    if sub_sample > 1:
        kw_dist = kw_dist[::sub_sample]
    if a_f > 1.0:  # apply filter only if <1
       for t in range(kw_dist.size(0)):
            if t > a_f:
                kw_dist[t] =  dist_array[t-a_f:t].mean()
    return kw_dist


# apply temporal filtering
def post_proc(kw_dist_array, sil_dist_array, labels, a_f=1.0, sub_sample=1):
    scores = []
    score_ntkwd = []
    score_kwd = []
    score_ntkwd_sil = []
    score_kwd_sil = []
    
    # apply temporal filtering and group the results according to the true category
    num_samples = len(labels)
    for i in range(num_samples):
        # temporal filtering
        kw_dist =  filter_signal(kw_dist_array[i], a_f=a_f, sub_sample=sub_sample )
        sil_dist =  filter_signal(sil_dist_array[i], a_f=a_f, sub_sample=sub_sample )

        # get the minimum value over time
        min_dist = kw_dist.abs().min()
        min_dist_sil = sil_dist.abs().min()
        scores.append(min_dist)

        if labels[i] == 'keyword':
            score_kwd.append(min_dist)
            score_kwd_sil.append(min_dist_sil)
        else:
            score_ntkwd.append(min_dist)
            score_ntkwd_sil.append(min_dist_sil)
    return score_kwd, score_ntkwd, score_kwd_sil, score_ntkwd_sil, scores

# inference on a set of data
def dataset_inference(  classifier, test_data, stride_ratio=0.125, \
        frame_window = 16000, pad = 16000,  noise_path=None, noise_snr=0, \
        random_start=False  ):

    # output lists
    result_kw = []  # distance from the 'keyword' prototype
    result_sil = [] # distance from the 'silence' prototype
    labels = []     # labels
    duration_neg = [] # durations of negative segments

    frame_stride =  int(frame_window*stride_ratio)  # number of pixel for the frame stride

    # load noise file if not None
    if noise_path is not None:
        noise, _ = librosa.load(noise_path, sr=16000)
        noise = noise[:frame_window]
        noise_power = compute_power(noise)

    for i in range(len(test_data)):
        item = test_data[i]
        samples, _ = librosa.load(item['audio_file_path'], sr=16000)
        if samples.shape[0] < frame_window:
            # pad with zeros if the sample is shorter than frame_window
            extra_pad = frame_window -  samples.shape[0]
        else:
            extra_pad = 0

        # collect results
        if item['is_hotword'] == 1:
            labels.append('keyword')
        else:
            labels.append('no_keyword')
            duration_neg.append(item['duration'])

        # additional pad with zeros
        samples = np.pad(samples, (pad,extra_pad+pad))
        if random_start:
            curr_idx = np.random.randint(1,frame_stride)
        else:
            curr_idx = 0

        
        #####################
        # compute the distance from the prototype window-by-window
        #####################
        
        # form a batch of windowed samples
        inf_segments = []        
        while curr_idx + frame_window <= len(samples):
            window = samples[curr_idx:curr_idx + frame_window]
        
            if noise_path is not None:
                # additive noise
                s_pow = compute_power(window)
                bg_vol = np.sqrt(s_pow/((10**noise_snr/10)*noise_power))
                window = window + (noise * bg_vol)

            new_sample = torch.Tensor(window)
            new_sample = new_sample.unsqueeze(0).unsqueeze(0)
            inf_segments.append(new_sample)
            curr_idx += frame_stride
        samples = torch.cat(inf_segments)

        # run batched inference
        p_y, target_ids = classifier.evaluate_batch(samples, ['keyword' for x in range(samples.size(0))], return_probas=False)
        speech_dist = p_y[:,0]  # L2 distances from the 'keyword' prototype
        sil_dist = p_y[:,1]     # L2 distances from the 'silence' prototype

        result_kw.append(speech_dist)
        result_sil.append(sil_dist)
    
    return result_kw, result_sil, labels, duration_neg

# calibration of the threshold
def calibration(classifier, pos_kwd, neg_kwd, noise_path=None, noise_files=[], \
                stride_ratio=0.125, pad=16000, frame_size = 16000, tau=0.5, alphas = [1]):
    
    # init
    final_a_f = 0
    final_thr_a_f = 0
    final_thr_no_a_f = 0
    best_gap = -100
    max_dist_vector = []

    # signal is padded w/ zeros
    pad_steps = int(pad/(frame_size*stride_ratio))

    # compute distances between the pos and neg calibration samples and the prototypes prototypes
    result_kw, result_sil, labels, _ = dataset_inference(classifier, pos_kwd+neg_kwd, \
            stride_ratio=stride_ratio, frame_window = frame_size, noise_path=None)
    
    # augment the calibration results w/ noisy samples
#    result_kw_snr, result_sil_snr, labels_snr, _ =  dataset_inference(classifier,pos_kwd, \
#            stride_ratio=stride_ratio, pad=pad, frame_size = frame_size, noise_path=noise_path, noise_files=noise_files, noise_snr=[0])
#    result_kw += result_kw_snr
#    result_sil += result_sil
#    labels += labels_snr

    # apply temporal filtering and check the alpha value leading to the largest gap
    for a_f in alphas:
        max_dist_pos = []
        max_dis_neg = []
        for i in range(len(labels)):
            speech_dist = filter_signal(result_kw[i], a_f=a_f)
            speech_dist = speech_dist[pad_steps:-pad_steps] # remove padding values
            max_dist = speech_dist.max().item()  # get minimum dstances from prototypes (speech_dist and max_dist are negatives values)
            if labels[i] == 'keyword':
                max_dist_pos.append(max_dist)
            else:
                max_dis_neg.append(max_dist)
        
        # take the positive/negative with the highest/lowest distance from the prototype
        max_dist_vector.append([min(max_dist_pos), max(max_dis_neg) ])
        gap = min(max_dist_pos) - max(max_dis_neg) 
        
        thr = -min(tau*gap, 0.5)+ min(max_dist_pos)
        print('- filter Lenght: {} | margin gap = {:.2f}, max_dis_neg= {:.2f}, max_dist_pos= {:.2f}'.format(\
            a_f, gap, -max(max_dis_neg),-min(max_dist_pos)))
        
        if a_f == 1:
            final_thr_no_a_f = thr

        if gap > best_gap:
            best_gap = gap
            final_thr_a_f = thr
            final_a_f = a_f

    return final_a_f, final_thr_a_f, final_thr_no_a_f, max_dist_vector

# test function on a set of data
def test(classifier, test_meta, stride_ratio=0.125, a_f=1, thr=None):

    # compute the distance scores of the test data with respect to the protototypes
    result_kw, result_sil, labels, duration_neg =  dataset_inference(classifier, test_meta, \
                stride_ratio=stride_ratio, frame_window = 16000, pad = 16000, noise_path=None)
    
    # apply temporal filtering and group the resutls with respect to the true categories 
    score_kwd, score_ntkwd, score_kwd_sil, score_ntkwd_sil, scores = post_proc(result_kw, result_sil, labels, a_f=a_f, sub_sample=1)
    pos_data = np.array(score_kwd)
    len_pos_data = len(pos_data)
    neg_data = np.array(score_ntkwd)
    results = { "scores": scores, 
                "labels":labels,
                "result_kw":result_kw,
                "perf":{},
            }
    
    # compute the metric: accuracy @ given far/h (values in far_h_list)
    far_h_list = [0.5, 1] # errors per hours
    for target_farh in far_h_list:

        # number of errors to achieve the target_farh
        time_neg = np.sum(duration_neg) / 3600
        target_error = int(target_farh *time_neg)
        
        # extract the threshold and compute the (optimal) accuracy
        optim_thr = np.sort(neg_data)[target_error]
        acc_opt = np.sum(pos_data < optim_thr) / len_pos_data
        frr_opt_perc = (1-acc_opt)*100
        results['perf']['thr_opt_'+str(int(target_farh*100))] = float(optim_thr)
        results['perf']['acc_opt_'+str(int(target_farh*100))] = acc_opt
        results['perf']['frr_opt_'+str(int(target_farh*100))] = frr_opt_perc

        print('-> Accuracy is {:.3f} at FARh = {} (optimal threshold: {:.3f})'.format(acc_opt, target_farh, optim_thr))

    # compute also the accuracy for a custom (given) threhsold
    if thr is not None:
        calib_thr = -thr
        acc_calib = np.sum(pos_data < calib_thr) / len_pos_data
        farh_calib = np.sum(neg_data < calib_thr) / time_neg
        frr_calib_perc = (1-acc_calib)*100
        print('-> With custom thr = {}, the accuracy is {} with farh = {}'.format(calib_thr, acc_calib, farh_calib))
        results['perf']['thr_calib'] = calib_thr
        results['perf']['acc_calib'] = acc_calib
        results['perf']['farh_calib'] = farh_calib
        results['perf']['frr_calib'] = frr_calib_perc

    return results

# segment a window of 16000 samples within a speech track with max power
def compute_start_snr(speech,frame_window = int(16000*0.025)):
    len_speech = len(speech)
    num_window = len_speech // frame_window
    len_frame = int(16000 / frame_window)
    max_pow = -1
    idx_start = -1
    for i in range(num_window-len_frame+1):
        speech_segment = speech[frame_window*i:frame_window*i+16000]
        power_speech_segment = np.power(speech_segment,2)
        power_speech_segment = np.sum(power_speech_segment)
        if power_speech_segment > max_pow:
            idx_start = i 
            max_pow = power_speech_segment
    return idx_start, max_pow

# segment samples for few-shot initialization
def get_samples(dataset, n_way, is_keyword=True, 
                different_speaker=False, # force the different speaker IDs for initialization
                frame_window =  int(16000*0.025)):
    buffer_data = []
    user_id_list = []
    new_ds = []
    for i in range(len(dataset)):
        if len(buffer_data)  == n_way:
            # exit the search
            break

        item = dataset[i]

        # skip samples if already in the user list AND different_speaker
        user_id = item['worker_id']
        if user_id in user_id_list and different_speaker: 
            continue

        samples, _ = librosa.load(item['audio_file_path'], sr=16000)
        if samples.shape[0] < 16000:
            extra_pad = 16000 -  samples.shape[0]
            samples = np.pad(samples, (0,extra_pad))

        # check if  is_keyword or not
        #print("***** Sample {} is a keyword: {}".format(i,item['is_hotword']))
        if item['is_hotword'] != 1: # negative speech 
            if not is_keyword:
                pass
                # debug print               
                #print('Sample {} used for NEGATIVE enrolling!'.format(i))
            else:
                continue

        else:   #keyword
            if is_keyword: 
                pass 
                # debug print               
                # print('Sample {} used for POSITIVE enrolling from user {} !'.format(i, user_id))
            else:
                continue
        
        # extract 1 sec of audio signal with max power from a utterance of variable lenght  
        idx_start , _ = compute_start_snr(samples, frame_window = frame_window)
        new_sample = samples[idx_start*frame_window:idx_start*frame_window+16000]
        buffer_data.append(new_sample)
        
        user_id_list.append(user_id)
        new_ds.append(item)
    return buffer_data, new_ds, user_id_list

# function to compare recognition stats before and after the incremental training    
def check(test_data, pos_id_list, pre_scores, pre_thr, post_scores, post_thr):
    pre_scores = np.array(pre_scores)
    post_scores = np.array(post_scores)
    tot_pos = 0
    counters = [0 for _ in range(8)]
    for i, item in enumerate(test_data):
        if item['is_hotword'] == 1:
            tot_pos += 1
            user_id = item['worker_id']
            if user_id in pos_id_list:
                if pre_scores[i] < pre_thr:
                    if post_scores[i] < post_thr:
                        counters[0] += 1
                    else:
                        counters[1] += 1
                else:
                    if post_scores[i] < post_thr:
                        counters[2] += 1
                    else:
                        counters[3] += 1
            else:
                if pre_scores[i] < pre_thr:
                    if post_scores[i] < post_thr:
                        counters[4] += 1
                    else:
                        counters[5] += 1
                else:
                    if post_scores[i] < post_thr:
                        counters[6] += 1
                    else:  
                        counters[7] += 1
    #print(counters, tot_pos, counters[4]/tot_pos, )
    return counters

##################################################
####### main script for self-leaning   ###########
##################################################
if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Self-learning Options')
    # model and dataset args
    parser.add_argument('--model_path', type=str, 
                        default='', help="Pretrained model (.pt) path")
    parser.add_argument('--dataset', type=str, default='heysnips',  
                            help="Type of dataset")
    parser.add_argument('--data_dir_pos', type=str, default='', 
                        help="Path to the positive voice dataset")
    parser.add_argument('--data_dir_neg', type=str, default='', 
                        help="Path to the negative voice dataset (set to data_dir_pos if empty)")
    parser.add_argument('--adapt_set_ratio', type=float, default=1.0,
                        help="Size of the adaptation partition (subset of the train set)")
    
    # parametrizing the labeler 
    parser.add_argument('--n_way', type=int, default=3, help="# examples per initialization")
    parser.add_argument('--pos_selflearn_thr', type=float, default=0.2,
                        help="Threshold for the positive samples during self labelling")
    parser.add_argument('--neg_selflearn_thr', type=float, default=0.9,
                        help="Threshold for the negative samples during self labelling")
    parser.add_argument('--frame_size', type=int, default=16000, help="Frame size in numner of samples")
    parser.add_argument('--step_size_ratio', type=float, default=0.125,
                        help="Step size in streaming inference as ratio of the frame size")    
    parser.add_argument('--tau', type=float, default=0.3,
                        help="Tau to calibrate the threshold (for inference, not pseudo-labeling)")
    parser.add_argument('--num_pos_batch', type=int, default=20,
                        help='number of positive samples inside a mini-batch for incremental learning')    
    parser.add_argument('--num_neg_batch', type=int, default=120,
                        help='number of negative samples inside a mini-batch for incremental learning')
    parser.add_argument('--min_pos_samples', type=int, default=10,
                        help='min number of pos samples to start incremental learning')
        
    # training options
    parser.add_argument('--train.epochs', type=int, default=4, metavar='NEPOCHS',
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                        help='optimization method (default: Adam)')
    parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--train.decay_every', type=int, default=20, metavar='LRDECAY',
                        help='number of epochs after which to decay the learning rate')
    parser.add_argument('--train.weight_decay', type=float, default=0.0, metavar='WD',
                        help="weight decay (default: 0.0)")
    parser.add_argument('--train.triplet_type', type=str, default='anchor_triplet', metavar='OPTIM',
                        help='options: [all_triplets , semirandom_neg, anchor_triplet (default)]')
    parser.add_argument('--train.force_silence_triplets', action='store_true', 
                        help="Force presence of silence samples (all zeros) oin the triplet")

    # log options
    parser.add_argument('--log.dirname', type=str, default='logs', 
            help="Result dirname")
    parser.add_argument('--log.results_json', type=str, default='log_continual_db_new_pers_final.json', 
            help="Filename for the json log")
    
    # other debug options
    parser.add_argument('--use_oracle', action='store_true', 
                        help="Use the oracle for labelling")
    parser.add_argument('--change_labels', type=float, default=0.0, 
                        help="Percentage of pos items missclassified")
    parser.add_argument('--optimal_sampling', action='store_true', 
                        help="Use the oracle for labelling")
    parser.add_argument('--num_continual_set', type=int, default=1,
                        help='number of incremental steps/partitions')
    parser.add_argument('--only_ok_pseudo_labels', action='store_true', 
                        help="Use the oracle for labelling")
    parser.add_argument('--also_ok_pos_pseudo_labels', action='store_true', 
                        help="Get anyway the pos pseudolabels, even if only_ok_pseudo_labels is True")
    parser.add_argument('--noise_path', type=str, default='', 
                        help="Path to the noise dataset")

    # get the options
    opt = vars(parser.parse_args())

    # fix the seed
    random.seed(71)

    # check arguments
    if opt['model_path'] == '':
        raise ValueError("Missing <model_path> argument!")
    if opt['data_dir_pos'] == '':
        raise ValueError("Missing <data_dir_pos> path for the {} banchmark!".format(opt['dataset']))
    if opt['data_dir_neg'] == '':
        opt['data_dir_neg'] = opt['data_dir_pos']

    # variables or fixed parameters
    n_way = opt['n_way']
    ORACLE = opt['use_oracle']
    N_POS_BATCH =opt['num_pos_batch']
    N_NEG_BATCH =opt['num_neg_batch']
    margin = 0.5
    alphas = [1, 2, 3, 4 , 5]
    results_global = {'settings': opt}

    # store logs
    dir_name = opt['log.dirname']
    json_log_file = opt['log.results_json']
    log_json_file = os.path.join(dir_name, json_log_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    print('Log file: ', log_json_file)
    if os.path.isfile(log_json_file):
        with open(log_json_file, 'r') as jsonFile:
            json_data = json.load(jsonFile)
        found = False
        for _, item in json_data.items():
            if item['settings'] == opt:
                found = True
        if found:
            print('Test with settings {} already logged'.format(opt))
            exit(0)


    ####################################################
    #### setup the dataset splits for the experiments
    ####################################################

    # prepare positive and negative partitions of train and test datasets
    args_data = { 'sample_rate':16000, 'out_dir': None }
    ds = WakeWordDataset(opt['data_dir_pos'], opt['data_dir_neg'], args_data, dataset=opt['dataset'] )
    print('Dataset is: ', ds.dataset)
    print('Number of train samples: {}.'.format(len(ds.train_meta)) )
    _  = ds.print_stats(ds.train_meta)
    print('Number of test samples: {}'.format( len(ds.test_meta)) )
    _  = ds.print_stats(ds.test_meta)

    # organize test set by splitting per-speaker data
    ds.get_person_splits(min_pos_sample=opt['min_pos_samples'], n_train_pos_samples = opt['n_way'], \
                         n_train_neg_samples = opt['n_way'], min_sample_duration_sec = 1)
    print('[Global Test data] Number of speakers is {} for a total of {} utterance ({} per speaker)'.format(ds.cnt_spk, ds.n_pos_samples,ds.n_pos_samples/ds.cnt_spk))
    print('[Global Test data] Number of negative utterance is {} for a total time of {} hours'.format(ds.n_neg_samples,ds.tot_time_neg/3600))

    # organize train set (fine tuning data for incremental learning)
    # -- we subsample the train set by <opt['adapt_set_ratio']> and split it into <num_continual_set> sessions
    # -- continual data is the set of new (unlabelled) data sessions
    n_speakers = ds.cnt_spk
    num_continual_set = 1
    adapt_set_ratio = opt['adapt_set_ratio']
    continual_data = ds.get_train_splits(adapt_set_ratio, num_continual_set=num_continual_set)

    # personalize the model w.r.t to speaker train data and then run the adaptation
    for ee in range(n_speakers):
        print('\n ***** Experiment with speaker: ', ee, ' *****')
        result_exp = {}    #  collect the log results of the present experiments

        ### assemble dataset for the current experiments
        # test set: per-speaker data for final test
        test_data = ds.pos_data_eval[ee]['test'] + ds.neg_data_eval['test']
        print('[Exp Test Data] Set of {} samples, of which {} are positives'.format(len(test_data), len(ds.pos_data_eval[ee]['test']) ) )
        # test set: per-speaker data for initialization (few-shot)
        len_init_set = len(ds.pos_data_eval[ee]['train'])
        print('[Exp Init Test Data] Number of item for the init set is {} '.format(len_init_set))
        # train data: rest of data used for incremental training (unlabelled)
        len_continual_set = [len(continual_data[i]) for i in range(num_continual_set)]
        for i in range(num_continual_set):
            print('[Exp Train Data]  Number of items for continual set #{}: {}'.format(i, len_continual_set[i]))
            cnt_kw, cnt_nkw = ds.print_stats(continual_data[i])
            result_exp['CL_set_samples_'+str(i)] = [cnt_kw, cnt_nkw]


        ##############################################################
        #### [Phase 1] Test the pretrained model on the test data

        # get few-shot data for initialization and pack to tensor (a.k.a. support set)
        segments_kwd, ds_pos, user_id_list = get_samples(ds.pos_data_eval[ee]['train'], n_way, is_keyword=True )
        segments_nokwd, ds_neg, _ =  get_samples(ds.neg_data_eval['train'], n_way, is_keyword=False )
        segments_kwd_th = torch.Tensor(segments_kwd).unsqueeze(1).unsqueeze(0)
        segments_nokwd_th = torch.Tensor(segments_nokwd).unsqueeze(1).unsqueeze(0)
        segments_silence = torch.zeros(segments_nokwd_th.size())
        support_samples = torch.cat([segments_kwd_th, segments_silence, segments_nokwd_th])
        class_list = ('keyword','silence', 'no_keyword')    # note: only 'keyword' is effectively used
        print('Size of support set:', support_samples.size())

        
        # load, initialize and test the model
        print('Initialize model {} using {} samples'.format(opt['model_path'], support_samples.size(1)))
        enc_model = torch.load(opt['model_path'])
        
        # use the encoder as the frozen backbone of the classifier
        classifier = NearestClassMean(backbone=enc_model, cuda=True)
        classifier.fit_batch_offline(support_samples, class_list, online_update = False, augument_proto = 0)
        classifier.muK = F.normalize(classifier.muK, p=2.0, dim=-1) # normalize prototype vectors

        # calibration step: finds the calibration thresholds (later used for pseudo-labeling)
        print('[Calibration] Calibrate the filter-lenght and the pseudo-labelling thresholds using the few-shot samples')
        a_f, thr, thr_noaf, max_dist_vector = calibration(classifier, ds_pos, ds_neg, \
                stride_ratio=opt['step_size_ratio'], pad=16000, frame_size = opt['frame_size'],\
                tau= opt['tau'], alphas=alphas)
        # print('-> Best filter lenght is {} with a threshold of {}. At a_f=1 the threshold is {}'.format(a_f, thr, thr_noaf))

        #get the threshold value in the optimal point
        i = np.where(np.array(alphas) == a_f)[0][0]
        pos_dist = max_dist_vector[i][0]
        neg_dist = max_dist_vector[i][1]
        gap = pos_dist - neg_dist
        
        # compute the pseudolabelling threshold
        POS_THR = abs(-min(opt['pos_selflearn_thr']*gap, 0.5)+ pos_dist )
        NEG_THR = abs(- opt['neg_selflearn_thr']*gap + pos_dist )
        print('-> Best filter lenght is {}. Positive and negative thresholds are: {:.2f} and {:.2f}'.format(a_f, POS_THR, NEG_THR ))
        
        result_exp['gap_step_pre'] = gap
        result_exp['POS_THR_pre'] = POS_THR
        result_exp['NEG_THR_pre'] = NEG_THR
        result_exp['a_f_pre'] = a_f

        # test on the current test set
        print('[Test before adaptation] start...')
        output_pre = test(classifier, test_data, stride_ratio=opt['step_size_ratio'], a_f=a_f, thr=thr)
        #print( output_pre['perf'])
        result_exp['res_pre'] = output_pre['perf']
        print('[Test before adaptation] Completed!')

        # multiple incremental training sessions (1 session == 1 new data set)
        # (only 1 session in our experiments!)

        for step in range(len(continual_data)):
            
            print('[Adaptation Step {}]'.format(step))

            ##############################################################
            #### [Phase 2] Assign pseudo labels to new samples inside the continual_data[step] list

            # reset the incremental training buffers
            pos_set = []
            neg_set = []


            ## using the true labels as pseudo-labels
            if ORACLE:  # True when --use_oracle option is used
                for item in continual_data[step]:
                    samples, _ = librosa.load( item['audio_file_path'], sr=16000)
                    if samples.shape[0] < opt['frame_size']:
                        extra_pad = opt['frame_size'] - samples.shape[0]
                        samples = np.pad(samples, (0,extra_pad))

                    if item['is_hotword'] == 1:
                        frame_window = int(opt['frame_size']*0.025) # different from stride. only for searching best SNR
                        idx_start , _ = compute_start_snr(samples, frame_window = frame_window)
                        samples = samples[idx_start*frame_window:idx_start*frame_window+opt['frame_size']]
                    else:
                        samples = np.pad(samples, (16000,16000))
                        start = random.randint(0, samples.shape[0] - opt['frame_size'])
                        samples = samples[start:start+opt['frame_size']]
                    
                    if samples.shape[0] < opt['frame_size']:
                        extra_pad = opt['frame_size'] - samples.shape[0]
                        samples = np.pad(samples, (0,extra_pad))  
                    
                    # fill the incremental learning buffers using the true labeles
                    samples = torch.Tensor(samples).unsqueeze(0).unsqueeze(0)
                    if item['is_hotword'] == 1:
                        pos_set.append(samples)
                    else: 
                        neg_set.append(samples)

                # [ablation option, nomrally 0] randomly invert some labels
                if opt['change_labels'] > 0:
                    random.shuffle(pos_set)
                    random.shuffle(neg_set)
                    len_invert = int(len(pos_set)*opt['change_labels'])
                    print('Going to change {}% of the positive labels ({} samples)'.format(int(opt['change_labels']*100),len_invert))        
                    ds_pos_change = pos_set[:len_invert]
                    ds_neg_change = neg_set[:len_invert]
                    pos_set = pos_set[len_invert:] + ds_neg_change
                    neg_set = neg_set[len_invert:] + ds_pos_change
                    random.shuffle(pos_set)
                    random.shuffle(neg_set)

            ## predict the pseudo-labels
            else:
                print('[Pseudo-labeling Task] start...')
                cnt_pos_ok = 0
                cnt_pos_nok = 0
                cnt_neg_ok = 0
                cnt_neg_nok = 0
                
                # compute distances with respect to the prototypes on the new unlabelled data in continual_data (order does not matter)
                # (batch comutation to speed-up the test, but same results as of as streaming processing)
                result_kw, result_sil, labels, _ =                      \
                    dataset_inference(classifier, continual_data[step], \
                                    stride_ratio=opt['step_size_ratio'],\
                                    frame_window = opt['frame_size'],   \
                                    pad = 16000, noise_path=None)
                
                print('[Pseudo-labeling Task] prediction completed')

                # label sample-by-sample based on the distance score from the prototypes
                num_samples = len(labels)
                for i in range(num_samples):
                    # temporal filtering
                    kw_dist =  filter_signal(result_kw[i], a_f=a_f, sub_sample=1 )

                    # get the minimum distance f
                    min_dist = kw_dist.abs().min()

                    if min_dist < POS_THR:
                        label_predicted = 'keyword'
                        argmin_dist = kw_dist.abs().argmin() - a_f + 1 # segment the new data stream considering the filter delay

                    elif  min_dist > NEG_THR:
                        # randomly sample a negative segment with a lenght as opt['frame_size']
                        # hyp: the system send some negative every now and then
                        argmin_dist = random.randint(0, len(kw_dist)-1)
                        label_predicted = 'no_keyword'
                    else:
                        # do not store the current data i for incremental learning
                        continue
                        
                    # get the data and pad with zeros
                    item = continual_data[step][i]
                    samples, _ = librosa.load(item['audio_file_path'], sr=16000)
                    if samples.shape[0] < opt['frame_size']:
                        extra_pad = opt['frame_size'] - samples.shape[0]
                    else:
                        extra_pad = 0
                    samples = np.pad(samples, (16000,extra_pad+16000))
                    len_samples = samples.shape[0]

                    if opt['optimal_sampling'] and item['is_hotword'] == 1:
                        # [ablation option] optimally extract an audio segment of lenght opt['frame_size'] from an audio stream of unspecified duration
                        frame_window = int(opt['frame_size']*0.025) # different from stride. only for searching best SNR
                        idx_start , _ = compute_start_snr(samples, frame_window = frame_window)
                        samples = samples[idx_start*frame_window:idx_start*frame_window+opt['frame_size']]                
                    else:
                        # segment the audio segment based on the filter response
                        curr_idx = int(max(argmin_dist,0) * opt['frame_size'] * opt['step_size_ratio'])
                        samples = samples[curr_idx:curr_idx + opt['frame_size']]
                    # finally convert to tensor
                    samples = torch.Tensor(samples).unsqueeze(0).unsqueeze(0)
                    
                    # assign pseudo labels and store new samples in the buffer
                    if opt['only_ok_pseudo_labels']: 
                        # [ablation option] negatives are forced to be correctly labelled
                        if label_predicted == 'keyword':
                            pos_set.append(samples)
                            if item['is_hotword'] == 1:
                                cnt_pos_ok += 1
                            elif opt['also_ok_pos_pseudo_labels']:
                                # [ablation option] not retain false positives
                                cnt_pos_nok += 1
                        else: 
                            if item['is_hotword'] != 1:
                                neg_set.append(samples)
                                cnt_neg_ok += 1
                    else:
                        # store paseudo-labeled samples in the memory buffers
                        if label_predicted == 'keyword':
                            pos_set.append(samples)
                            if item['is_hotword'] == 1:
                                cnt_pos_ok += 1
                            else:
                                cnt_pos_nok += 1
                        else: 
                            #neg_set.append(samples*random.uniform(0, 1))
                            neg_set.append(samples)
                            if item['is_hotword'] == 1:
                                cnt_neg_nok += 1
                            else:
                                cnt_neg_ok += 1
                
                print('[Pseudo-labeling Task] labels assignment completed')
                print('-> pseudo-positive samples correctly labeled: {} (true positives)'.format(cnt_pos_ok))
                print('-> pseudo-positive samples NOT correctly labeled: {} (false positives)'.format(cnt_pos_nok))
                print('-> pseudo-negative samples correctly labeled: {} (true negatives)'.format(cnt_neg_ok))
                print('-> pseudo-negative samples NOT correctly labeled: {} (false negatives)'.format(cnt_neg_nok))

                result_exp['cnt_pos_ok_step_'+str(step)] = cnt_pos_ok
                result_exp['cnt_pos_nok_step_'+str(step)] = cnt_pos_nok
                result_exp['cnt_neg_ok_step_'+str(step)] = cnt_neg_ok
                result_exp['cnt_neg_nok_step_'+str(step)] = cnt_neg_nok

            ##############################################################
            #### [Phase 3] Incrementally train the encoder

            #incremental buffers
            len_pos = len(pos_set)
            len_neg = len(neg_set)

            # set the encoder to training mode
            enc_model.train()

            # setup the optimizer            
            optim_method = getattr(optim, opt['train.optim_method'])
            optim_config = { 'lr': opt['train.learning_rate'],
                                'weight_decay': opt['train.weight_decay'] }
            optimizer = optim_method(enc_model.parameters(), **optim_config)

            # number of batch depends on the total number of pseudo-positive samples (len_pos)
            n_batch = len_pos // N_POS_BATCH

            # pick N_NEG_BATCH from len_neg
            n_neg_batch = len_neg
            if n_neg_batch > N_NEG_BATCH:
                n_neg_batch = N_NEG_BATCH

            if n_batch == 0:
                # not enough new positive data for training
                print('[Incremental Training] No training: new pseudo-positive data lower than the treshold {}'.format(N_POS_BATCH))    

            else:
                print('[Incremental Training] start training for {} epochs with {} mini-batches per epoch'.format(opt['train.epochs'], n_batch))    

                for ep in range(opt['train.epochs']):
                    random.shuffle(pos_set)
                    for i in range(n_batch):
                        # get the new positive samples of the minibatch 
                        samples = pos_set[i*N_POS_BATCH:(i+1)*N_POS_BATCH]
                        
                        #get the new negative samples of the minibatch
                        random.shuffle(neg_set)
                        neg_samples = neg_set[:n_neg_batch] 
                        
                        # compose the minibatch
                        if opt['train.triplet_type'] == 'anchor_triplet': #default
                            # add few-shot user-provided utterances 
                            seg = [torch.Tensor(t).unsqueeze(0).unsqueeze(0) for t in segments_kwd]
                            samples = torch.cat(seg+samples+neg_samples, dim=0)
                        else:
                            samples = torch.cat(samples+neg_samples, dim=0)
                        
                        # [ablation option, default is False] ass silence (all zeros) samples to the minibatch
                        if opt['train.force_silence_triplets']:
                            samples = torch.cat([samples, torch.zeros(samples.size()[1:]).unsqueeze(0)], dim=0)


                        # get the embeddings samples (forwars step)
                        x = samples.cuda()
                        zq = enc_model.get_embeddings(x)

                        #compute loss
                        if opt['train.triplet_type'] == 'anchor_triplet': #default
                            # triplets index (ix,iy,iz), where ix is a user-provided sample, iy is a new pseudo-positive and iz is a new pseudo negative
                            anchor_positives = [[n_way+j,x] for j in range(N_POS_BATCH) for x in range(n_way)]
                            negative_indices = n_way+N_POS_BATCH+np.where(range(n_neg_batch))[0]
                            triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                                                        for neg_ind in negative_indices]

                        elif opt['train.triplet_type'] == 'all_triplets':
                            # same as before but ix and iy are all combinations of the posotive samples
                            anchor_positives = list(combinations(range(N_POS_BATCH), 2))  # All anchor-positive pairs
                            negative_indices = N_POS_BATCH+np.where(range(n_neg_batch))[0]
                            triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                                                        for neg_ind in negative_indices]

                        elif opt['train.triplet_type'] == 'semirandom_neg':
                            # only considers triplets with high-margin (optimize the hardest examples)
                            anchor_positives = list(combinations(range(N_POS_BATCH), 2))  # All anchor-positive pairs
                            negative_indices = N_POS_BATCH+np.where(range(n_neg_batch))[0]
                            triplets = []
                            for anchor_positive in anchor_positives:
                                ap_distances = (zq[anchor_positive[0]] - zq[anchor_positive[1]]).pow(2).sum()
                                loss_values = [ 1.0 + ap_distances - (zq[anchor_positive[0]] - - zq[neg_ind]).pow(2).sum() for neg_ind in negative_indices]
                                loss_values = np.array([value.item() for i,value in enumerate(loss_values) ])
                                hard_negatives = np.where(loss_values > 0)[0]
                                rr = np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None
                                if rr is not None:
                                    triplets.append([anchor_positive[0], anchor_positive[1], rr])
                                #print(anchor_positive[0], anchor_positive[1], hard_negatives, rr)
                            if len(triplets) == 0:
                                triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

                        else: # option not recognized
                            raise ValueError("Type of triplet loss {} not recognized!".format(opt['train.triplet_type'] ))

                        
                        if opt['train.force_silence_triplets']:
                            neg_ind = samples.size(0) - 1
                            #print('The silence index is : ', neg_ind)
                            for anchor_positive in anchor_positives:
                                triplets.append( [anchor_positive[0], anchor_positive[1], neg_ind] )
                                                        
                        # get the triplet tensors
                        triplets = torch.LongTensor(np.array(triplets))
                        
                        # optimizer update!!
                        optimizer.zero_grad()        
                        # compute triplet loss
                        ap_distances = (zq[triplets[:, 0]] - zq[triplets[:, 1]]).pow(2).sum(1) 
                        an_distances = (zq[triplets[:, 0]] - zq[triplets[:, 2]]).pow(2).sum(1)
                        #print(ap_distances.mean().item(), an_distances.mean().item())
                        loss = F.relu(ap_distances - an_distances + margin).mean()
                        
                        loss.backward()
                        optimizer.step()
                print('[Incremental Training] completed')    

            ##############################################################
            #### [Phase 4] Test after the update
            print('[Test after incremental training #{}] start...'.format(step))

            # freeze the new model and setup the classifier using the few-shot known samples
            enc_model.eval()
            classifier = NearestClassMean(backbone=enc_model, cuda=True)
            classifier.fit_batch_offline(support_samples, class_list, online_update = False, augument_proto = 0)
            classifier.muK = F.normalize(classifier.muK, p=2.0, dim=-1)

            # model calibration
            print('[Calibration] Re-Calibrate the filter-lenght after the incremental learning step')
            a_f, thr, thr_noaf, max_dist_vector = calibration(classifier, ds_pos, ds_neg, \
                    stride_ratio=opt['step_size_ratio'], pad=16000, frame_size = opt['frame_size'],  \
                    tau=opt['tau'], alphas=alphas)

            #get the threshold value in the optimal point
            i = np.where(np.array(alphas) == a_f)[0][0]
            pos_dist = max_dist_vector[i][0]
            neg_dist = max_dist_vector[i][1]
            gap = pos_dist - neg_dist        
            POS_THR = abs(-min(opt['pos_selflearn_thr']*gap, 0.5)+ pos_dist )
            NEG_THR = abs(- opt['neg_selflearn_thr']*gap + pos_dist )
            print('-> Best filter lenght is {}. Positive and negative thresholds are: {:.2f} and {:.2f}'.format(a_f, POS_THR, NEG_THR ))
            
            result_exp['gap_step_'+str(step)] = gap
            result_exp['POS_THR_step_'+str(step)] = POS_THR
            result_exp['NEG_THR_step_'+str(step)] = NEG_THR
            result_exp['a_f_step_'+str(step)] = a_f

            # test after incremental learning
            print('[Test after adaptation #{}] start...'.format(step))
            output_post = test(classifier, test_data, stride_ratio=opt['step_size_ratio'], a_f=a_f, thr=thr)
            result_exp['res_step_'+str(step)] = output_post['perf']
            print('[Test before adaptation] Completed!')

        result_exp['res_checks_'+str(step)] = check(test_data, user_id_list, output_pre['scores'],output_pre['perf']['thr_opt_50'], output_post['scores'], output_post['perf']['thr_opt_50'])
        results_global[str(ee)] = result_exp

    exit(0)
    print(results_global)

    # average results and store final stats
    log_json_file = os.path.join(dir_name, json_log_file)
    print(log_json_file)
    if os.path.isfile(log_json_file):
        with open(log_json_file, 'r') as jsonFile:
            json_data = json.load(jsonFile)
        next_id = len(json_data)
    else:
        next_id = 0
        json_data = {}

    json_data[str(next_id)] = results_global
    with open(log_json_file, "w") as jsonFile:
        json.dump(json_data, jsonFile)
