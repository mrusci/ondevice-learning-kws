import os
from functools import partial
import torch
import hashlib
import math
import os.path
import random
import re
import glob
import time

import pandas as pd
import soundfile as sf
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchnet.transform import compose
from torchnet.dataset import ListDataset, TransformDataset
import torchaudio
import torch.nn.functional as F


from .data_utils import SetDataset

class EpisodicFixedBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, fixed_silence_unknown = False):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.sampling = [torch.randperm(self.n_classes)[:self.n_way] for i in range(self.n_episodes)]

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield self.sampling[i]



class MSWCDataset:
    def __init__(self, data_dir, MSWCtype, cuda, args):
        self.sample_rate = args['sample_rate']
        self.clip_duration_ms = args['clip_duration'] 
        self.window_size_ms = args['window_size']
        self.window_stride_ms = args['window_stride']
        self.n_mfcc = args['n_mfcc']
        self.feature_bin_count = args['num_features']
        self.foreground_volume = args['foreground_volume']
        self.time_shift_ms = args['time_shift']
        self.desired_samples = int(self.sample_rate * self.clip_duration_ms / 1000)

        # main data dir
        self.data_dir = data_dir

        # add noise
        self.use_background = args['include_noise']
        self.background_volume = args['noise_snr']
        self.background_frequency= args['noise_frequency']
        if self.use_background:
            print('Load background data')
            self.background_data = self.load_background_data()
        else:
            self.background_data = []
        self.mfcc = self.build_mfcc_extractor()
        
        #try if can I include cuda here
        #cuda=True
        self.cuda = cuda
        if cuda:
            self.mfcc.cuda()
        
        #remove data from the GSC10 dataset
        avoid_words = ['yes','no','up','down','left','right','on','off','stop','go']
        
        min_words = 0
        balance = False
        if MSWCtype == 'MSWC350':
            n_classes = 350
            balance = True
        elif MSWCtype == 'MSWC200':
            n_classes = 200
            balance = True
        elif MSWCtype == 'MSWC500':
            n_classes = 500
            balance = True
        elif MSWCtype == 'MSWC500U':
            n_classes = 500
        elif MSWCtype == 'MSWC1000U':
            n_classes = 1000
        elif MSWCtype == 'MSWC5000U':
            n_classes = 5000
        elif MSWCtype == 'MSWC10000U':
            n_classes = 10000
        elif MSWCtype == 'MSWC15000U':
            n_classes = 15000
        elif MSWCtype == 'MSWC-LargeU':
            n_classes = None
            min_words = 2
            balance = False
        else:
            raise ValueError('Partition {} not supported for MSWC dataset'.format(MSWCtype) )

        self.generate_data_dictionary(n_classes, min_words, balance, avoid_words)
        self.max_class = len(self.wanted_words)
        
        self.word_to_index = {}
        for i,word in enumerate(self.wanted_words):
            self.word_to_index[word] = i

    def generate_data_dictionary(self, n_classes=None, min_words=0, balance = True, avoid_words = None):

        # Prepare data sets
        self.data_set = {'training': [], 'validation': [], 'testing': []}
        wanted_words = []        

        for split in ["training","validation", "testing"]:
            
            # parse the right file
            if split == 'training':
                split_name = 'train'
            elif split == 'validation':
                split_name = 'dev'
            elif split == 'testing':
                split_name = 'test'
            df = pd.read_csv(self.data_dir+"en_"+split_name+".csv")
            parse_word = {}
            # compute the number of samples per class
            for word in df['WORD']:
                if word in avoid_words:
                    continue
            
                if word in parse_word.keys():
                    parse_word[word] +=1
                else:
                    parse_word[word] = 1
            sorted_parse_word = sorted(parse_word.items(), key=lambda kv: kv[1], reverse=True)

            tot_samples = 0
            min_samples = len(df['WORD'])
            max_samples = 0 
            if split == "training":
                if n_classes is None:
                    n_classes = len(sorted_parse_word)

                wanted_words = []
                for item in sorted_parse_word[:n_classes]:
                    word = item[0]
                    n_occur = item[1]
                    if n_occur > min_words:
                        wanted_words.append(word)
                        min_samples = min(min_samples, n_occur)
                        max_samples = max(max_samples, n_occur)
                        tot_samples += n_occur
                self.wanted_words = wanted_words
                self.n_classes = len(wanted_words)

            else:
                for item in sorted_parse_word:
                    word = item[0]
                    n_occur = item[1]
                    if word in wanted_words:
                        min_samples = min(min_samples, n_occur)
                        max_samples = max(max_samples, n_occur)
                        tot_samples += n_occur

            n_classes = len(wanted_words)

            print("[Split: {}] {} classes with a number of words between {} and {}. Total of {} samples [avg of {} per class]"\
             .format(split,n_classes,min_samples,max_samples, tot_samples,tot_samples/n_classes ))

            # build the dict dataset split
            word_list = df['WORD']
            file_list = df['LINK']
            spk_list = df['SPEAKER']
            samples_per_words = {}
            for i,word in enumerate(word_list):
                if word in wanted_words:
                    wav_path = file_list[i].replace(".opus",".wav")
                    speaker_id = spk_list[i]
                    if balance:
                        if word in samples_per_words.keys():
                            if samples_per_words[word] < min_samples:
                                samples_per_words[word] += 1
                                self.data_set[split].append({'label': word, 'file': wav_path})
                        else:
                            samples_per_words[word] = 1
                            self.data_set[split].append({'label': word, 'file': wav_path})                            
                    else:
                        self.data_set[split].append({'label': word, 'file': wav_path})
            
            del df


    def dataset_filter_class(self, dslist, classes):
    # FIXME: not the fastest way but works
        filtered_ds = []
        extra_ds = []
        for k,item in enumerate(dslist):
            label = item['label']
            if label in classes:
                filtered_ds.append(item)
            else:
                extra_ds.append(item)
        return filtered_ds, extra_ds
    
    def get_transform_dataset(self, file_dict, classes, filters=None):
        # file dict include is [{ 'label': LABEL_str, 'file': file_path, 'speaker': spkr_id}, .. ]
        # classes is a list of classes
        transforms = compose([
                partial(self.load_audio, 'file', 'label', 'data'),
                #partial(self.adjust_volume, 'data'),
                partial(self.shift_and_pad, 'data', 'file'),
                partial(self.mix_background, self.use_background,'data'),
                #partial(self.extract_features, 'data', 'feat'),
                partial(self.label_to_idx, 'label', 'label_idx')

        ])
        file_dict, rest_of_data = self.dataset_filter_class(file_dict, classes)
        ls_ds = ListDataset(file_dict)
        ts_ds = TransformDataset(ls_ds, transforms)
        
        return ts_ds, rest_of_data

    def get_episodic_fixed_sampler(self, num_classes,  n_way, n_episodes, fixed_silence_unknown = False):
        return EpisodicFixedBatchSampler(num_classes, n_way, n_episodes, fixed_silence_unknown = fixed_silence_unknown)    
    
    def get_episodic_dataloader(self, set_index, n_way, n_samples, n_episodes, sampler='episodic', 
        include_silence=True, include_unknown=True, unique_speaker=False):

        if sampler == 'episodic':
            sampler = self.get_episodic_fixed_sampler(len(self.wanted_words),  
                        n_way, n_episodes)

        dl_list=[]        
        if set_index in ['training', 'validation', 'testing']:
            dataset = self.data_set[set_index]
            for k, keyword in enumerate(self.wanted_words):
                # debug print 
#                if k % 100 == 0:
#                    print('Train set == ', k)

                ts_ds, dataset = self.get_transform_dataset(dataset, [keyword])

                if n_samples <= 0:
                    n_samples = len(ts_ds)
                
                dl = torch.utils.data.DataLoader(ts_ds, batch_size=n_samples,  
                        shuffle=True, num_workers=0)
                dl_list.append(dl)

            ds = SetDataset(dl_list)
            data_loader_params = dict(batch_sampler = sampler, num_workers=8, 
                    pin_memory=not self.cuda)  
            dl = torch.utils.data.DataLoader(ds, **data_loader_params)
        else:
            raise ValueError("Set index = {} in episodic dataset is not correct.".format(set_index))

        return dl
    
    
    def get_iid_dataloader(self, split, batch_size, unique_speaker=False):
             
        ts_ds, _ = self.get_transform_dataset(self.data_set[split], self.wanted_words)
        if split =='training':
            batch_size = batch_size
        else: 
            batch_size = 1
        dl = torch.utils.data.DataLoader(ts_ds, batch_size=batch_size, pin_memory=not self.cuda, shuffle=True, num_workers=8)

        return dl
    
    def num_classes(self):
        return len(self.wanted_words)
        
    def label_to_idx(self, k, key_out, d):
        label_index = self.word_to_index[d[k]]
        d[key_out] = torch.LongTensor([label_index]).squeeze()

        return d

    def mix_background(self, use_background, k, d):
        if use_background and len(self.background_data) > 0: # add background noise as data augumentation
            foreground = d[k]
            background_index = np.random.randint(len(self.background_data))
            background_samples = self.background_data[background_index]
            if len(background_samples) <= self.desired_samples:
                raise ValueError(
                    'Background sample is too short! Need more than %d'
                    ' samples but only %d were found' %
                    (self.desired_samples, len(background_samples)))
            background_offset = np.random.randint(
                0, len(background_samples) - self.desired_samples)
            background_clipped = background_samples[background_offset:(
                background_offset + self.desired_samples)]
            background_reshaped = background_clipped.reshape([1, self.desired_samples])
        
            if np.random.uniform(0, 1) < self.background_frequency:
                bg_snr = np.random.uniform(0, self.background_volume)
                s_pow = foreground.pow(2).sum()
                n_pow = background_reshaped.pow(2).sum()
                bg_vol = (s_pow/((10**bg_snr/10)*n_pow)).sqrt().item()
            else:
                bg_vol = 0

            background_mul = background_reshaped * bg_vol
            background_add = background_mul + foreground
            background_clamped = torch.clamp(background_add, -1.0, 1.0)
            d[k] = background_clamped

        return d
    
    def extract_features(self, k, key_out, d):
        # moved to the model

        if self.cuda:
            d_in = d[k].cuda()
        else:
            d_in = d[k]
        features = self.mfcc(d_in)[0] # just one channel
        features = torch.narrow(features, 0, 0, self.feature_bin_count)
        features = features.T # f x t -> t x f
        d[key_out] = torch.unsqueeze(features,0)

        return d

    def load_background_data(self):
        background_path = os.path.join(self.data_dir, '../noise/', '*.wav')
        background_data = []
        if self.use_background:
            for wav_path in glob.glob(background_path):
                bg_sound, bg_sr = torchaudio.load(wav_path)
                background_data.append(bg_sound.flatten())
        return background_data
    
    def build_mfcc_extractor(self):
        # moved to the model

        def next_power_of_2(x):  
            return 1 if x == 0 else 2**math.ceil(math.log2(x))
        
        frame_len = self.window_size_ms / 1000
        stride = self.window_stride_ms / 1000
        n_fft = next_power_of_2(frame_len*self.sample_rate)

        mfcc = torchaudio.transforms.MFCC(self.sample_rate,
                                        n_mfcc=self.n_mfcc ,
                                        log_mels = True,
                                        melkwargs={
                                            'win_length': int(frame_len*self.sample_rate),
                                            'hop_length' : int(stride*self.sample_rate),
                                            'n_fft' : int(frame_len*self.sample_rate),
                                            "n_mels": self.n_mfcc,
                                            "power": 2,
                                            "center": False                                         
                                        }
                                         )
        return mfcc

    def shift_and_pad(self, key, key_path, d):
        audio = d[key]

        audio =  audio * self.foreground_volume
        time_shift = int((self.time_shift_ms * self.sample_rate) / 1000)
        if time_shift > 0:
            time_shift_amount = np.random.randint(-time_shift, time_shift)
        else:
            time_shift_amount = 0
        
        if time_shift_amount > 0:
            time_shift_padding = (time_shift_amount, 0)
            time_shift_offset = 0
        else:
            time_shift_padding = (0, -time_shift_amount)
            time_shift_offset = -time_shift_amount
        
        
        # Ensure data length is equal to the number of desired samples
        audio_len = audio.size(1)
        if audio_len < self.desired_samples:
            pad = (0,self.desired_samples-audio_len)
            audio=F.pad(audio, pad, 'constant', 0) 

        padded_foreground = F.pad(audio, time_shift_padding, 'constant', 0)
        sliced_foreground = torch.narrow(padded_foreground, 1, time_shift_offset, self.desired_samples)
        d[key] = sliced_foreground

        return d

    
    def load_audio(self, key_path, key_label, out_field, d):
        #format d struct: {'label': 'stop', 'file': '../../data/speech_commands/GSC/stop/879a2b38_nohash_3.wav', 'speaker': '879a2b38'}
        filepath = self.data_dir + 'clips_wav/'+ d[key_path]
        sound, sr = torchaudio.load(filepath=filepath, normalize=True)
        if sr != self.sample_rate:
            sound = torchaudio.functional.resample(sound, sr, self.sample_rate)
        d[out_field] = sound

        return d



    
