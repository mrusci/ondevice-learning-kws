import json
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Self-learning Options')
# model and dataset args
parser.add_argument('--model_path', type=str, 
                    default='', help="Path to the model pt under test")
parser.add_argument('--dataset', type=str, default='', 
                    help="Path to the sensor dataset")
parser.add_argument('--log_name', type=str, default='log_continual_db_new_pers.json', 
                    help="Log file name")

opt = vars(parser.parse_args())



# check arguments
if opt['model_path'] == '':
    raise ValueError("Missing <model_path> argument!")
if opt['dataset'] not in ['heysnips','heysnapdragon']:
    raise ValueError("<dataset> is wrong!")



log_json_file = os.path.join(opt['model_path'],opt['log_name'])

with open(log_json_file, 'r') as jsonFile:
    json_data = json.load(jsonFile)

def print_settings(sett_values, print_keys=False):
    v = ''
    for kk,vv in sett_values.items():
        if print_keys:
            vv = kk
        v += str(vv) + '\t'
    print(v)

# print table labels
aa = ''
for item in range(20):
    aa  += 'acc50_pre_'+ str(item)+ '\t'
for item in range(20):
    aa  += 'acc50_post_'+str(item)+ '\t'
print(
    'dataset\t',
    'use_oracle\t',
    'triplet_type\t',
    'change_labels\t',
    'adapt_set_ratio\t',
    'pos_selflearn_thr\t',
    'neg_selflearn_thr\t',
    'pos_tot_mean\t',
    'neg_tot_mean\t',
    'pos_samples_mean\t',
    'pos_wrong_mean\t',
    'neg_samples_mean\t',
    'neg_wrong_mean\t',
    'a_f_pre_mean\t',
    'a_f_inc_mean\t',
    'acc_opt_50_pre_mean\t',
    'acc_opt_50_inc_mean\t',
    'acc_opt_50_pre_std\t',
    'acc_opt_50_inc_std\t',
    'acc_opt_50_pre_median\t',
    'acc_opt_50_inc_median\t',
    'acc_calib_pre_mean\t',
    'farh_calib_pre_mean\t',
    'acc_calib_inc_mean\t',
    'farh_calib_inc_mean\t',
    aa,
)

# average and print experiment data 
for key, exp in json_data.items(): 
    settings = exp['settings']
    pos_selflearn_thr = settings['pos_selflearn_thr']
    neg_selflearn_thr = settings['neg_selflearn_thr']
    step_size_ratio = settings['step_size_ratio']
    use_oracle = settings['use_oracle']
    change_labels = settings['change_labels']
    num_continual_set = settings['num_continual_set']
    adapt_set_ratio = settings['adapt_set_ratio']
    init_set_ratio = settings['init_set_ratio']
    num_pos_batch = settings['num_pos_batch']
    num_neg_batch = settings['num_neg_batch']
    num_experiments = settings['num_experiments']
    triplet_type = settings['train.triplet_type']

    # filters
    if (not use_oracle) and (triplet_type != 'anchor_triplet'): # discard other results
        continue
    if not settings['dataset'] == opt['dataset']:
        continue
    if 'train.force_silence_triplets' in settings.keys():
        if settings['train.force_silence_triplets'] is True:
            continue 

    acc_opt_50_pre_list = []
    acc_opt_50_inc_list = []
    acc_calib_pre_list = []
    farh_calib_pre_list = []
    acc_calib_inc_list = []
    farh_calib_inc_list = []
    
    a_f_pre_list = []
    a_f_inc_list = []
    pos_tot_list = []
    neg_tot_list = []       
    pos_samples_list = []
    pos_wrong_list = []
    neg_samples_list = []
    neg_wrong_list = []

    check_list = [ []for i in range(8)]
    per_spk_acc_list = {}
    per_spk_acc_list_pre = {}

    for key2, ee in exp.items(): 
        if 'settings' in key2:
            continue
        last = num_continual_set -1


        #get pre
        gap_step_pre = ee['gap_step_pre']
        POS_THR_pre = ee['POS_THR_pre']
        NEG_THR_pre = ee['NEG_THR_pre']
        a_f_pre = ee['a_f_pre']
        res_pre = ee['res_pre']
        acc_opt_50_pre = res_pre['acc_opt_50']
        acc_opt_100_pre = res_pre['acc_opt_100']
        acc_calib_pre = res_pre['acc_calib']
        farh_calib_pre = res_pre['farh_calib']

        # get post
        gap_step_inc = ee['gap_step_'+str(last)]
        POS_THR_inc = ee['POS_THR_step_'+str(last)]
        NEG_THR_inc = ee['NEG_THR_step_'+str(last)]
        a_f_inc = ee['a_f_step_'+str(last)]
        res_inc = ee['res_step_'+str(last)]
        acc_opt_50_inc = res_inc['acc_opt_50']
        acc_opt_100_inc = res_inc['acc_opt_100']
        acc_calib_inc = res_inc['acc_calib']
        farh_calib_inc = res_inc['farh_calib']

        per_spk_acc_list[key2]= acc_opt_50_inc
        per_spk_acc_list_pre[key2] = acc_opt_50_pre 

        checks = ee['res_checks_'+str(last)]
        tot = np.sum(checks)
        for i in range(8):
            check_list[i].append( checks[i] / tot )

        # data
        CL_set_samples = ee['CL_set_samples_'+str(last)]
        if not use_oracle:
            cnt_pos_ok = ee['cnt_pos_ok_step_'+str(last)]
            cnt_pos_nok = ee['cnt_pos_nok_step_'+str(last)]
            cnt_neg_ok = ee['cnt_neg_ok_step_'+str(last)]
            cnt_neg_nok = ee['cnt_neg_nok_step_'+str(last)]
            pos_samples = cnt_pos_ok + cnt_pos_nok
            if pos_samples == 0:
                pos_wrong = 0
            else:
                pos_wrong = cnt_pos_nok / pos_samples
            neg_samples = cnt_neg_ok + cnt_neg_nok
            neg_wrong = cnt_neg_nok / neg_samples
            #print(CL_set_samples,cnt_pos_ok,cnt_pos_nok,cnt_neg_ok,cnt_neg_nok, acc_opt_50_pre, acc_opt_50_inc, a_f_inc)
        else:
            pos_samples = CL_set_samples[0]
            pos_wrong = 0 / pos_samples
            neg_samples = CL_set_samples[1]
            neg_wrong = 0 / neg_samples
            #print(CL_set_samples, acc_opt_50_pre, acc_opt_50_inc, a_f_inc)

        acc_opt_50_pre_list.append(acc_opt_50_pre)
        acc_opt_50_inc_list.append(acc_opt_50_inc)

        acc_calib_pre_list.append(acc_calib_pre)
        farh_calib_pre_list.append(farh_calib_pre)
        acc_calib_inc_list.append(acc_calib_inc)
        farh_calib_inc_list.append(farh_calib_inc)

        #print('After:', acc_opt_50_pre, acc_opt_50_inc)
        a_f_pre_list.append(a_f_pre)
        a_f_inc_list.append(a_f_inc)

        pos_tot_list.append(CL_set_samples[0])
        neg_tot_list.append(CL_set_samples[1])      
        pos_samples_list.append(pos_samples)
        pos_wrong_list.append(pos_wrong)
        neg_samples_list.append(neg_samples)
        neg_wrong_list.append(neg_wrong)
    
    # avg
    acc_opt_50_pre_mean = np.mean(acc_opt_50_pre_list)
    acc_opt_50_inc_mean = np.mean(acc_opt_50_inc_list)
    acc_opt_50_pre_std = np.std(acc_opt_50_pre_list)
    acc_opt_50_inc_std = np.std(acc_opt_50_inc_list)
    acc_opt_50_pre_median = np.median(acc_opt_50_pre_list)
    acc_opt_50_inc_median = np.median(acc_opt_50_inc_list)

    acc_calib_pre_mean = np.mean(acc_calib_pre_list)
    farh_calib_pre_mean = np.mean(farh_calib_pre_list)
    acc_calib_inc_mean = np.mean(acc_calib_inc_list)
    farh_calib_inc_mean = np.mean(farh_calib_inc_list)
    
    check_mean = [np.mean(item) for item in check_list ]

    a_f_pre_mean = np.mean(a_f_pre_list)
    a_f_inc_mean = np.mean(a_f_inc_list)

    pos_tot_mean = np.mean(pos_tot_list)
    neg_tot_mean = np.mean(neg_tot_list)
    pos_samples_mean = np.mean(pos_samples_list)
    pos_wrong_mean = np.mean(pos_wrong_list)
    neg_samples_mean = np.mean(neg_samples_list)
    neg_wrong_mean = np.mean(neg_wrong_list)


    aa = ''
    for item in acc_opt_50_pre_list:
        aa  += str(item)+ '\t'
    for item in acc_opt_50_inc_list:
        aa  += str(item)+ '\t'

    for item in check_mean:
        aa  += str(item)+ '\t'

    print(
        settings['dataset'], '\t',
        use_oracle, '\t',
        triplet_type, '\t',
        change_labels, '\t',
        adapt_set_ratio, '\t',
        pos_selflearn_thr, '\t',
        neg_selflearn_thr, '\t',
        pos_tot_mean, '\t',
        neg_tot_mean, '\t',
        pos_samples_mean, '\t',
        pos_wrong_mean, '\t',
        neg_samples_mean, '\t',
        neg_wrong_mean, '\t',
        a_f_pre_mean, '\t',
        a_f_inc_mean, '\t',
        acc_opt_50_pre_mean, '\t',
        acc_opt_50_inc_mean,'\t',
        acc_opt_50_pre_std, '\t',
        acc_opt_50_inc_std,'\t',
        acc_opt_50_pre_median,'\t',
        acc_opt_50_inc_median,'\t',
        acc_calib_pre_mean,'\t',
        farh_calib_pre_mean,'\t',
        acc_calib_inc_mean,'\t',
        farh_calib_inc_mean,'\t',
        aa,
    )

    