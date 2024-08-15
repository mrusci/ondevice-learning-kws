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
from classifiers.NCM import NearestClassMean
from scipy.io.wavfile import read
import torch.optim as optim


##################################################
####### test functions
##################################################

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
        kw_dist =  filter_signal(kw_dist_array[i], a_f=a_f, sub_sample=sub_sample )
        min_dist = kw_dist.abs().min()
        scores.append(min_dist)
        if labels[i] == 'keyword':
            score_kwd.append(min_dist)
        else:
            score_ntkwd.append(min_dist)
    return score_kwd, score_ntkwd, score_kwd_sil, score_ntkwd_sil, scores

def dataset_inference(classifier, test_data, labels, data_dir, stride_ratio=0.125, \
                    frame_window = 16000 ):
    
    frame_stride =  int(frame_window*stride_ratio)
    result_kw = []
    result_sil = []
    duration_neg = []

    for i in range(len(test_data)):

        # load data
        item = test_data[i]
        samples = np.load(os.path.join(data_dir, item)).astype(float)
        samples = samples / (2**24) # the normalization value depends from the acquisition system
    
        if labels[i] == 'no_keyword':
            time = samples.shape[0] / 16000
            duration_neg.append(time)

        # concatenate window frames into a data batch
        curr_idx = 0
        inf_segments = []       
        while curr_idx + frame_window <= len(samples):
            window = samples[curr_idx:curr_idx + frame_window]
            window = window - np.mean(window)
            new_sample = torch.Tensor(window)
            new_sample = new_sample.unsqueeze(0).unsqueeze(0)
            inf_segments.append(new_sample)
            curr_idx += frame_stride
        samples = torch.cat(inf_segments)

        # compute distance from a data batch
        p_y, target_ids = classifier.evaluate_batch(samples, ['keyword' for x in range(samples.size(0))], return_probas=False)
        speech_dist = p_y[:,0]
        result_kw.append(speech_dist)

    return result_kw, result_sil, labels, duration_neg

# calibration of the threshold
def calibration(classifier, pos_kwd, neg_kwd, data_dir, noise_path=None, noise_files=[], \
                stride_ratio=0.125, pad=16000, frame_size = 16000, tau=0.5, alphas = [1]):
    
    pad_steps = int(pad/(frame_size*stride_ratio))

    # test
    labels = ['keyword' for _ in range(len(pos_kwd))] + ['no_keyword' for _ in range(len(neg_kwd))]
    result_kw, result_sil, labels, _ =  dataset_inference(classifier, pos_kwd+neg_kwd, labels, data_dir, \
            stride_ratio=stride_ratio, frame_window = frame_size)
    
    #compute thresholds
    final_a_f = 0
    final_thr_a_f = 0
    final_thr_no_a_f = 0
    best_gap = -100
    max_dist_vector = []
    for a_f in alphas:
        max_dist_pos = []
        max_dis_neg = []
        for i in range(len(labels)):
            speech_dist = filter_signal(result_kw[i], a_f=a_f)
            speech_dist = speech_dist[pad_steps:-pad_steps]
            max_dist = speech_dist.max().item()
            if labels[i] == 'keyword':
                max_dist_pos.append(max_dist)
            else:
                max_dis_neg.append(max_dist)
            l = str(i)+('k' if labels[i] == 'keyword' else '')

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

def test(classifier, test_meta, labels, data_dir, stride_ratio=0.125, a_f=1, thr=None):
    # compute the distance scores of the test data with respect to the protototypes
    result_kw, result_sil, labels, duration_neg =  dataset_inference(classifier, test_meta, labels, data_dir, \
                                                                     stride_ratio=stride_ratio, frame_window = 16000)
    
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
    far_h_list = [0.5, 1]
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

##################################################
## main script for self-leaning on real data  ####
##################################################
if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Self-learning Options')
    # model and dataset args
    parser.add_argument('--model_path', type=str, 
                        default='', help="Path to the model pt under test") #
    parser.add_argument('--data_dir', type=str, default='', 
                        help="Path to the sensor dataset") #
    parser.add_argument('--ds_config_file', type=str, default='dscnnl_fp16', 
                        help="Path to the configuration file for the experiment") #
    
    # parametrizing the labeler 
    parser.add_argument('--use_oracle', action='store_true', 
                        help="Use the oracle for labelling") #
    parser.add_argument('--pos_selflearn_thr', type=float, default=0.2,
                        help="Threshold for the positive samples during self labelling")
    parser.add_argument('--neg_selflearn_thr', type=float, default=0.9,
                        help="Threshold for the negative samples during self labelling")
    parser.add_argument('--frame_size', type=int, default=16000, help="Frame size in numer of samples")
    parser.add_argument('--num_pos_batch', type=int, default=20,
                        help='number of positive samples inside a mini-batch for incremental learning')  #  
    parser.add_argument('--num_neg_batch', type=int, default=120,
                        help='number of negative samples inside a mini-batch for incremental learning') #
    parser.add_argument('--step_size_ratio', type=float, default=0.125,
                        help="Step size in streaming inference as ratio of the frame size")  
    parser.add_argument('--tau', type=float, default=0.3,
                        help="Tau to calibrate the threshold")

    # experiments options
    parser.add_argument('--num_experiments', type=int, default=10,
                        help='number of runs')
    parser.add_argument('--only_ok_pseudo_labels', action='store_true', 
                        help="Use the oracle for labelling")    
    parser.add_argument('--also_ok_pos_pseudo_labels', action='store_true', 
                        help="Get anyway the pos pseudolabels, even if only_ok_pseudo_labels is True")

    # training options
    parser.add_argument('--train.epochs', type=int, default=4, metavar='NEPOCHS',
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                        help='optimization method (default: Adam)')
    parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--train.decay_every', type=int, default=20, metavar='LRDECAY',
                        help='number of epochs after which to decay the learning rate')
    parser.add_argument('--train.weight_decay', type=float, default=0.0, metavar='WD',
                        help="weight decay (default: 0.0)")
    parser.add_argument('--train.triplet_type', type=str, default='anchor_triplet', metavar='OPTIM',
                        help='options: [anchor_triplet (default), semirandom_neg, all_triplets]')

    
    # get the options
    opt = vars(parser.parse_args())

    # fix the seed
    random.seed(71)

    # check arguments
    if opt['model_path'] == '':
        raise ValueError("Missing <model_path> argument!")
    if opt['data_dir'] == '':
        raise ValueError("Missing <data_dir> path!")


    # variables or fixed parameters
    N_POS_BATCH =opt['num_pos_batch']
    N_NEG_BATCH =opt['num_neg_batch']
    alphas = [1, 2, 3, 4 , 5]
    margin = 0.5
    results_global = {}

    # get config file for the experiments
    with open(os.path.join(opt['data_dir'], 'logs/', opt['ds_config_file'])) as f:
        ds_file_list = json.load(f)
    n_speakers = len(ds_file_list['test'].keys()) - 1
    speaker_list = [x for x in ds_file_list['test'].keys() if 'spk' in x ]
    print('Start Experiment with {} speakers: {}.'.format(n_speakers, speaker_list))

    # per-speaker test
    for ee in speaker_list:
        id = int(ee.replace('spk_',''))
        for i_run in range(opt['num_experiments']):

            print('\n ***** Experiment with speaker {} (id: {}) | Run: {}/{}  *****'.format(ee, id, i_run+1, opt['num_experiments'] ))

            result_exp = {}    

            # load the model
            enc_model = torch.load(opt['model_path'])
            classifier = NearestClassMean(backbone=enc_model, cuda=True)
            init_path_pos = os.path.join(opt['data_dir'], 'dst/', ee+'_initwav')
            init_path_neg = os.path.join(opt['data_dir'], 'dst/', 'neg_initwav')
            segments_kwd = []
            for filewav in  os.listdir(init_path_pos):
                fs, a = read(os.path.join(init_path_pos, filewav))
                arr = np.array(a).astype(float) / 2**8 # the normalization factor is needed to align the range of the wav file and the raw data
                arr = arr - np.mean(arr)    # remove mean
                #print(arr, np.max(arr), np.min(arr))
                segments_kwd.append(arr)
            n_way = len(segments_kwd)
            segments_kwd_th = torch.Tensor(segments_kwd).unsqueeze(1).unsqueeze(0)
            print('Initialize model {} using {} samples'.format(opt['model_path'], n_way))

            classifier.fit_batch_offline(segments_kwd_th, ['keyword'], online_update = False, augument_proto = 0)
            classifier.muK = F.normalize(classifier.muK, p=2.0, dim=-1)


            # claibration data to compute the threshold (corresponds to the few shot)
            calibration_data_pos = ds_file_list['test'][ee]['init']
            calibration_data_neg = ds_file_list['test']['neg']['init']

            ##############################################################
            #### [Step 1] Test the pretrained model on the test raw data. 
            
            # get the file name (not the scores computed on-device)
            ds_pos = [x for x in calibration_data_pos ]
            ds_neg = [x for x in calibration_data_neg ]

            # calibration step: finds the calibration thresholds (later used for pseudo-labeling)
            print('[Calibration] Calibrate the filter-lenght and the pseudo-labelling thresholds using the few-shot samples')
            a_f, thr, thr_noaf, max_dist_vector = calibration(classifier, ds_pos, ds_neg, os.path.join(opt['data_dir'], 'dst/'), \
                                            stride_ratio=opt['step_size_ratio'], pad=16000, frame_size = opt['frame_size'],\
                                            tau= opt['tau'], alphas=alphas)

            #get the threshold value in the optimal point
            i = np.where(np.array(alphas) == a_f)[0][0]
            pos_dist = max_dist_vector[i][0]
            neg_dist = max_dist_vector[i][1]
            gap = pos_dist - neg_dist
        
            # compute the pseudolabelling threshold
            POS_THR = abs(-min(opt['pos_selflearn_thr']*gap, 0.5)+ pos_dist )
            NEG_THR = abs(- opt['neg_selflearn_thr']*gap + pos_dist )
            print('-> Best filter lenght is {}. Positive and negative thresholds are: {:.2f} and {:.2f}'.format(a_f, POS_THR, NEG_THR ))

            
            # test
            print('[Test before adaptation on Raw Data] start...')
            test_data_pos = ds_file_list['test'][ee]['test']
            test_data_neg = ds_file_list['test']['neg']['test']
            test_data = [x for x in test_data_pos ] + [x for x in test_data_neg ]
            labels = ['keyword' for _ in range(len(test_data_pos))] + ['no_keyword' for _ in range(len(test_data_neg))]
            output_pre = test(classifier, test_data, labels, os.path.join(opt['data_dir'], 'dst/'), stride_ratio=opt['step_size_ratio'], a_f=a_f, thr=thr)
            result_exp['res_pre'] = output_pre['perf']
            print('[Test before adaptation] Completed!')


            ##############################################################
            #### [Step 2] Test the pretrained model on GAP 
            #### Compute the accuracy based on the logs from the devices
            #### The obtained results are compared with the scores computed in the precedent step
            #### to assess the difference between full-precision vs. int8 inference

            # (note 1) te cablibration and the test are performend based on the measurement returned by the board
            # at time-stamps k*opt['step_size_ratio']. In our setup, the first 7 scores are obtained by concatenating parts
            # of consecutive utterances. Therefore, we discard these measurements in the following.
            
            # calibration step: finds the calibration thresholds (later used for pseudo-labeling)
            final_a_f = 0
            final_thr_a_f = 0
            final_thr_no_a_f = 0
            best_gap = -100
            tau = 0.5
            max_dist_vector = []
            for a_f in alphas:
                max_dist_pos = []
                max_dis_neg = []

                # compute max distance wrt positive samples
                for item in calibration_data_pos:
                    scores = np.array(calibration_data_pos[item])
                    result_kw = torch.Tensor(scores[:, id])
                    speech_dist = filter_signal(result_kw, a_f=a_f)
                    speech_dist = speech_dist[7+a_f:]   # skip the first 7 scores (see note 1 above)
                    max_dist = speech_dist.min().item()
                    max_dist_pos.append(max_dist)

                # compute max distance wrt negative samples
                for item in calibration_data_neg:
                    scores = np.array(calibration_data_neg[item])
                    result_kw = torch.Tensor(scores[:, id])
                    speech_dist = filter_signal(result_kw, a_f=a_f)
                    speech_dist = speech_dist[7+a_f:]   # skip the first 7 scores (see note 1 above)
                    max_dist = speech_dist.min().item()
                    max_dis_neg.append(max_dist)

                max_dist_vector.append([max(max_dist_pos), min(max_dis_neg) ])
                gap = min(max_dis_neg) - max(max_dist_pos)
                thr = -min(tau*gap, 0.5)+ min(max_dist_pos)
                print('- filter Lenght: {} | margin gap = {:.2f}, max_dis_neg= {:.2f}, max_dist_pos= {:.2f}'.format(\
                    a_f, gap, -max(max_dis_neg), -min(max_dist_pos)))
                
                if gap > best_gap:
                    best_gap = gap
                    final_thr_a_f = thr
                    final_a_f = a_f
            
            #get the threshold value in the optimal point
            i = np.where(np.array(alphas) == final_a_f)[0][0]
            pos_dist = max_dist_vector[i][0]
            neg_dist = max_dist_vector[i][1]
            gap = neg_dist - pos_dist
            
            # compute the pseudolabelling threshold
            POS_THR = abs(+min(opt['pos_selflearn_thr']*gap, 0.5)+ pos_dist )
            NEG_THR = abs(+ opt['neg_selflearn_thr']*gap + pos_dist )
            print('Positive and negative thresholds are: {:.2f} and {:.2f} for a filter of lenght {}'.format(POS_THR, NEG_THR, final_a_f))
            print('-> Best filter lenght is {}. Positive and negative thresholds are: {:.2f} and {:.2f}'.format(final_a_f, POS_THR, NEG_THR ))

            result_exp['gap_step_pre'] = gap
            result_exp['POS_THR_pre'] = POS_THR
            result_exp['NEG_THR_pre'] = NEG_THR
            result_exp['a_f_pre'] = final_a_f

            # test on-device
            print('[Test before adaptation on on-device scores computed w/ int8 inference] start...')
            
            # read scores from the test set
            test_data_pos = ds_file_list['test'][ee]['test']
            test_data_neg = ds_file_list['test']['neg']['test']
            labels = ['keyword' for _ in range(len(test_data_pos))] + ['no_keyword' for _ in range(len(test_data_neg))] 
            result_kw = []
            for test_data in [test_data_pos, test_data_neg]:
                for item in test_data:
                    scores = np.array(test_data[item])
                    result_kw.append(torch.Tensor(scores[7:, id])) # discard first 7 scores (see note 1 above)
        
            #fitler the scores and compute max/min
            scores = []
            score_ntkwd = []
            score_kwd = []
            duration_neg = []
            num_samples = len(labels)
            for i in range(num_samples):
                kw_dist =  filter_signal(result_kw[i], a_f=final_a_f, sub_sample=1 )
                min_dist = kw_dist.abs().min()
                scores.append(min_dist)
                if labels[i] == 'keyword':
                    score_kwd.append(min_dist)
                else:
                    score_ntkwd.append(min_dist)
                    duration_neg.append(len(result_kw[i])*1/8)  #a samples is computed every 1/8 seconds
            
            time_neg = np.sum(duration_neg) / 3600
            print('Total Time of negative samples is {} hours'.format(time_neg))
            pos_data = np.array(score_kwd)
            len_pos_data = len(test_data_pos)
            neg_data = np.array(score_ntkwd)

            # compute the metric: accuracy @ given far/h (values in far_h_list)
            results = { "scores": scores, 
                        "labels":labels,
                        "result_kw":result_kw,
                        "perf":{},
                    }
            far_h_list = [0.5, 1]
            for target_farh in far_h_list:
                # compute target error

                # number of errors to achieve the target_farh
                target_error = int(target_farh *time_neg)
                #print('thr: ', target_farh, target_error)
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
                print('At thr {:.3f}: accuracy of {:.3f} with farh = {}'.format(calib_thr, acc_calib, farh_calib))
                results['perf']['thr_calib'] = calib_thr
                results['perf']['acc_calib'] = acc_calib
                results['perf']['farh_calib'] = farh_calib
                results['perf']['frr_calib'] = frr_calib_perc
            
            result_exp['res_pre_gap'] = results['perf']
            print('[Test before adaptation] Completed!')

            ##############################################################
            #### [Step 3] Adaptation: assign pseudo-labels based on the scores computed on GAP
            print('[Adaptation Step] start....')
            
            # reset the incremental training buffers
            pos_set = []
            neg_set = []

            # adaptation set
            train_data_pos = ds_file_list['train']['pos']
            train_data_neg = ds_file_list['train']['neg']

            # intermediate results
            labels = []
            item_list = []
            result_kw_array = []

            #read the scores from the logs
            for item in train_data_pos:
                scores = np.array(train_data_pos[item])
                result_kw = torch.Tensor(scores[7:, id]) # discard first 7 scores (see note 1 above)
                result_kw_array.append(result_kw)
                labels.append('keyword')
                item_list.append(item)

            for item in train_data_neg:
                scores = np.array(train_data_neg[item])
                result_kw = torch.Tensor(scores[7:, id]) # discard first 7 scores (see note 1 above)
                result_kw_array.append(result_kw)
                labels.append('no_keyword')
                item_list.append(item)

            ## using the true labels as pseudo-labels
            if opt['use_oracle']:

                num_samples = len(labels)
                for i in range(num_samples):
                    kw_dist =  filter_signal(result_kw_array[i], a_f=final_a_f, sub_sample=1 )
                    min_dist = kw_dist.abs().min()
                    if labels[i] == 'keyword':
                        label_predicted = 'keyword'
                        # segmentation based on the distance from the prototype
                        argmin_dist = kw_dist.abs().argmin() - final_a_f + 1 
                    else:
                        # take a random value 
                        # hyp: the system send some negative every now and then
                        argmin_dist = random.randint(0,len(kw_dist)-1)
                        label_predicted = 'no_keyword'

                    # read raw data
                    file_path_dst = os.path.join(opt['data_dir'], 'dst/', item_list[i])
                    if os.path.isfile(file_path_dst):
                        samples = np.load(file_path_dst).astype(float)
                        samples = samples / (2**24) # the normalization value depends from the acquisition system
                    else:
                        continue

                    curr_idx = int(max(argmin_dist,0) * opt['frame_size'] * opt['step_size_ratio'])
                    samples = samples[curr_idx:curr_idx + opt['frame_size']]

                    # fill the incremental learning buffers using the true labeles
                    samples = torch.Tensor(samples).unsqueeze(0).unsqueeze(0)
                    samples = samples - samples.mean()
                    if labels[i] == 'keyword':
                        pos_set.append(samples)
                    else: 
                        neg_set.append(samples)

            ## predict the pseudo-labels
            else:
                print('[Pseudo-labeling Task] start...')
                cnt_pos_ok = 0
                cnt_pos_nok = 0
                cnt_neg_ok = 0
                cnt_neg_nok = 0

                num_samples = len(labels)
                for i in range(num_samples):

                    # perform psudo-labeling based on the thresholds
                    kw_dist =  filter_signal(result_kw_array[i], a_f=final_a_f, sub_sample=1 )
                    min_dist = kw_dist.abs().min()
                    if min_dist < POS_THR:
                        label_predicted = 'keyword'
                        # take the prediction
                        argmin_dist = kw_dist.abs().argmin() - final_a_f + 1 

                    elif  min_dist > NEG_THR:
                        # take a random value 
                        # hyp: the system send some negative every now and then
                        argmin_dist = random.randint(0,len(kw_dist)-1)
                        label_predicted = 'no_keyword'
                    else:
                        continue
                    
                    # get the data
                    file_path_dst = os.path.join(opt['data_dir'], 'dst/', item_list[i])
                    if os.path.isfile(file_path_dst):
                        samples = np.load(file_path_dst).astype(float)
                        samples = samples / (2**24) # the normalization value depends from the acquisition system
                    else:
                        continue
        
                    len_samples = samples.shape[0]
                    curr_idx = int(max(argmin_dist,0) * opt['frame_size'] * opt['step_size_ratio'])
                    samples = samples[curr_idx:curr_idx + opt['frame_size']]


                    # assign pseudo labels and store new samples in the buffer
                    samples = torch.Tensor(samples).unsqueeze(0).unsqueeze(0)
                    samples = samples - samples.mean()

                    if opt['only_ok_pseudo_labels']: 
                        # [ablation option] negatives are forced to be correctly labelled
                        if label_predicted == 'keyword':
                            if labels[i] == 'keyword':
                                cnt_pos_ok += 1
                                pos_set.append(samples)
                            elif opt['also_ok_pos_pseudo_labels']:
                                cnt_pos_nok += 1
                                pos_set.append(samples)
                        else: 
                            if labels[i] == 'no_keyword':
                                neg_set.append(samples)
                                cnt_neg_ok += 1
                    else:
                        if label_predicted == 'keyword':
                            pos_set.append(samples)
                            if labels[i] == 'keyword':
                                cnt_pos_ok += 1
                            else:
                                cnt_pos_nok += 1
                        else: 
                            neg_set.append(samples)
                            if labels[i] == 'keyword':
                                cnt_neg_nok += 1
                            else:
                                cnt_neg_ok += 1

                print('[Pseudo-labeling Task] labels assignment completed')
                print('-> pseudo-positive samples correctly labeled: {} (true positives)'.format(cnt_pos_ok))
                print('-> pseudo-positive samples NOT correctly labeled: {} (false positives)'.format(cnt_pos_nok))
                print('-> pseudo-negative samples correctly labeled: {} (true negatives)'.format(cnt_neg_ok))
                print('-> pseudo-negative samples NOT correctly labeled: {} (false negatives)'.format(cnt_neg_nok))



                result_exp['cnt_pos_ok_step_0'] = cnt_pos_ok
                result_exp['cnt_pos_nok_step_0'] = cnt_pos_nok
                result_exp['cnt_neg_ok_step_0'] = cnt_neg_ok
                result_exp['cnt_neg_nok_step_0'] = cnt_neg_nok


            ##############################################################
            #### [Step 4] Incrementally train the encoder

            #incremental buffers
            len_pos = len(pos_set)
            len_neg = len(neg_set)
            print('[Incremental Training] Summary: Buffer of new data include {} positives and {} negatives'.format(len_pos,len_neg))    

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
            if n_neg_batch>N_NEG_BATCH:
                n_neg_batch = N_NEG_BATCH

            result_exp['adapt'] = False
            if n_batch == 0:
                # not enough new positive data for training
                print('[Incremental Training] No training: new pseudo-positive data lower than the treshold {}'.format(N_POS_BATCH))    
            else:
                print('[Incremental Training] start training for {} epochs with {} mini-batches per epoch'.format(opt['train.epochs'], n_batch))    

                result_exp['adapt'] = True
                for ep in range(opt['train.epochs']):
                    random.shuffle(pos_set)

                    for i in range(n_batch):                        
                        
                        # get the new positive samples of the minibatch 
                        samples = pos_set[i*N_POS_BATCH:(i+1)*N_POS_BATCH]

                        #get the new negative samples of the minibatch
                        random.shuffle(neg_set)
                        neg_samples = neg_set[:n_neg_batch] 
                            
                        # compose the minibatch
                        if opt['train.triplet_type'] == 'anchor_triplet':
                            seg = [torch.Tensor(t).unsqueeze(0).unsqueeze(0) for t in segments_kwd]
                            samples = torch.cat(seg+samples+neg_samples, dim=0)
                        else:
                            samples = torch.cat(samples+neg_samples, dim=0)

                        # get the embeddings samples (forwars step)
                        x = samples.cuda()
                        zq = enc_model.get_embeddings(x)

                        #compute loss
                        if opt['train.triplet_type'] == 'anchor_triplet':
                            anchor_positives = [[n_way+j,x] for j in range(N_POS_BATCH) for x in range(n_way)]
                            negative_indices = n_way+N_POS_BATCH+np.where(range(n_neg_batch))[0]
                            triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                                                        for neg_ind in negative_indices]
                        
                        elif opt['train.triplet_type'] == 'all_triplets':
                            anchor_positives = list(combinations(range(N_POS_BATCH), 2))  # All anchor-positive pairs
                            negative_indices = N_POS_BATCH+np.where(range(n_neg_batch))[0]
                            triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                                                        for neg_ind in negative_indices]
                        
                        elif opt['train.triplet_type'] == 'semirandom_neg':
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

                        # get the triplet tensors
                        triplets = torch.LongTensor(np.array(triplets))
                        
                        # optimizer update!!
                        optimizer.zero_grad()        
                        # compute triplet loss
                        ap_distances = (zq[triplets[:, 0]] - zq[triplets[:, 1]]).pow(2).sum(1) 
                        an_distances = (zq[triplets[:, 0]] - zq[triplets[:, 2]]).pow(2).sum(1)
                        loss = F.relu(ap_distances - an_distances + margin).mean()
                        
                        loss.backward()
                        optimizer.step()

            print('[Incremental Training] completed')    


            ##############################################################
            #### [Step 5] Test after the update
            print('[Test after incremental training] start...')  

            # freeze the new model and setup the classifier using the few-shot known samples
            enc_model.eval()
            classifier = NearestClassMean(backbone=enc_model, cuda=True)
            classifier.fit_batch_offline(segments_kwd_th, ['keyword'], online_update = False, augument_proto = 0)
            classifier.muK = F.normalize(classifier.muK, p=2.0, dim=-1)

            # model calibration
            print('[Calibration] Re-Calibrate the filter-lenght after the incremental learning step')
            a_f, thr, thr_noaf, max_dist_vector = calibration(classifier, ds_pos, ds_neg, os.path.join(opt['data_dir'], 'dst/'), \
                                        stride_ratio=opt['step_size_ratio'], pad=16000, frame_size = opt['frame_size'],\
                                        tau= opt['tau'], alphas=alphas)
            print('Best filter factor is {} with a threshold of {}. At a_f=1 the threshold is {}'.format(a_f, thr, thr_noaf))

            #get the threshold value in the optimal point
            i = np.where(np.array(alphas) == a_f)[0][0]
            pos_dist = max_dist_vector[i][0]
            neg_dist = max_dist_vector[i][1]
            gap = pos_dist - neg_dist        
            POS_THR = abs(-min(opt['pos_selflearn_thr']*gap, 0.5)+ pos_dist )
            NEG_THR = abs(- opt['neg_selflearn_thr']*gap + pos_dist )        
            print('-> Best filter lenght is {}. Positive and negative thresholds are: {:.2f} and {:.2f}'.format(a_f, POS_THR, NEG_THR ))

            result_exp['gap_step_0'] = gap
            result_exp['POS_THR_step_0'] = POS_THR
            result_exp['NEG_THR_step_0'] = NEG_THR
            result_exp['a_f_step_0'] = a_f

            # test after incremental learning
            print('[Test after adaptation] start...')
            test_data_pos = ds_file_list['test'][ee]['test']
            test_data_neg = ds_file_list['test']['neg']['test']
            test_data = [x for x in test_data_pos ] + [x for x in test_data_neg ]
            labels = ['keyword' for _ in range(len(test_data_pos))] + ['no_keyword' for _ in range(len(test_data_neg))]
            output_post = test(classifier, test_data, labels, os.path.join(opt['data_dir'], 'dst/'), stride_ratio=opt['step_size_ratio'], a_f=a_f, thr=thr)
            result_exp['res_step_0'] = output_post['perf']        
            print('[Test after adaptation] Completed!')


            #append log
            if str(ee) not in list(results_global.keys()):
                results_global[str(ee)] = []
            results_global[str(ee)].append(result_exp)


    print('*********************************************')
    print('******** Final Summary **********')
    print('*********************************************')
    print('Model File: ', opt['model_path'])
    print('Score File: ', opt['ds_config_file'] )
    print('\n* Settings:')
    print('Positive Threshold: ', opt['pos_selflearn_thr'])
    print('Negative Treshold: ', opt['neg_selflearn_thr'] )
    print('Loss Type: ', opt['train.triplet_type'])
    print('# pos samples in a minibatch: ', opt['num_pos_batch'])
    print('# neg samples in a minibatch: ', opt['num_neg_batch'])
    print('#epochs: ', opt['train.epochs'])
    
    
    # compute results per speaker
    print('\n*  Final Results (per speaker)')
    acc_list = []
    acc_pre_list = []
    acc_pre_gap_list = []
    diff_list = []
    diff_gap_list = []
    if opt['use_oracle']:
        print( "speaker \tAcc. Pre-adapt \tAcc. Pre-adapt on GAP \tDiff Pre-adapt vs. GAP \tadapt \tAcc. Post-adapt \tdiff \tStdDev Pre-adapt \tStdDev Pre-adapt on GAP \tStdDev Post-adapt ")
    else:
        print( "speaker \tAcc. Pre-adapt \tAcc. Pre-adapt on GAP \tDiff Pre-adapt vs. GAP \tadapt \tAcc. Post-adapt \tdiff \tStdDev Pre-adapt \tStdDev Pre-adapt on GAP \tStdDev Post-adapt \t cnt_pos_ok \tcnt_pos_nok \tcnt_neg_ok \tcnt_neg_nok\t")           
    for ee in speaker_list:
        output_pre = []
        output_pre_gap = []
        output_post = []
        adapt = []
        cnt_pos_ok = []
        cnt_pos_nok = []
        cnt_neg_ok = []
        cnt_neg_nok = []

        for i_run in range(opt['num_experiments']):
            result_exp = results_global[str(ee)][i_run]
            output_pre.append( result_exp['res_pre']['acc_opt_50']) # pre-adapt accuracy 
            output_pre_gap.append( result_exp['res_pre_gap']['acc_opt_50']) # pre-adapt accuracy computed on GAP (not for comparison) 
            output_post.append( result_exp['res_step_0']['acc_opt_50']) # post-adapt accuracy 
            adapt.append( result_exp['adapt'])
            if not opt['use_oracle']:
                cnt_pos_ok.append( result_exp['cnt_pos_ok_step_0'])
                cnt_pos_nok.append( result_exp['cnt_pos_nok_step_0'])
                cnt_neg_ok.append( result_exp['cnt_neg_ok_step_0'])
                cnt_neg_nok.append( result_exp['cnt_neg_nok_step_0'])

        std_output_pre = np.std(output_pre)
        std_output_pre_gap  = np.std(output_pre_gap)
        std_output_post = np.std(output_post)

        output_pre = np.mean(output_pre)
        output_pre_gap  = np.mean(output_pre_gap)
        output_post = np.mean(output_post)
        adapt = np.mean(adapt)
        
        if not opt['use_oracle']:
            cnt_pos_ok = np.mean(cnt_pos_ok)
            cnt_pos_nok = np.mean(cnt_pos_nok)
            cnt_neg_ok = np.mean(cnt_neg_ok)
            cnt_neg_nok = np.mean(cnt_neg_nok)

        diff = output_post - output_pre
        diff_gap = output_pre_gap - output_pre


        acc_list.append(output_post)
        acc_pre_list.append(output_pre)
        acc_pre_gap_list.append(output_pre_gap)
        diff_list.append(diff)
        diff_gap_list.append(diff_gap)
        if opt['use_oracle']:
            print( ee, "\t{:.2f}\t".format(output_pre), "{:.2f}\t".format(output_pre_gap), "{:.2f}\t".format(diff_gap),  \
                 adapt, "\t{:.2f}\t".format(output_post), "{:.2f}\t".format(diff), \
                    "{:.2f}\t{:.2f}\t{:.2f}\t".format(std_output_pre,std_output_pre_gap,std_output_post ) )
        else:
            print( ee, "\t{:.2f}\t".format(output_pre), "{:.2f}\t".format(output_pre_gap), "{:.2f}\t".format(diff_gap),  \
                 adapt, "\t{:.2f}\t".format(output_post), "{:.2f}\t".format(diff), \
                    "{:.2f}\t{:.2f}\t{:.2f}\t".format(std_output_pre,std_output_pre_gap,std_output_post ),\
                 "{}\t".format(cnt_pos_ok), "{}\t".format(cnt_pos_nok), "{}\t".format(cnt_neg_ok), "{}\t".format(cnt_neg_nok))           

    print('\n*  Final Results (summary)')
    print('Pre-adapt Accuracy\t{}\t with a std dev of\t{}'.format(np.mean(np.array(acc_pre_list)), np.std(np.array(acc_pre_list)) )) 
    print('Pre-adapt Accuracy on GAP\t{}\t with a std dev of\t{}'.format(np.mean(np.array(acc_pre_gap_list)),np.std(np.array(acc_pre_gap_list)) ))
    print('Post-adapt Accuracy\t{}\t with a std dev of\t{}'.format( np.mean(np.array(acc_list)), np.std(np.array(acc_list))))
    print('Increment\t{}\t with a std dev of\t{}'.format(np.mean(np.array(diff_list)), np.std(np.array(diff_list))))


