# code from Cristi


import os
from functools import partial
import glob
import hashlib
import math
import os.path
import random
import re
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchnet.transform import compose
from torchnet.dataset import ListDataset, TransformDataset
import torchaudio
import torch.nn.functional as F

from .data_utils import SetDataset



# settings of Google Speech Commands
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
BACKGROUND_NOISE_LABEL = '_background_noise_'
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 1
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 0
RANDOM_SEED = 59185

            
class EpisodicFixedBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, fixed_silence_unknown = False, include_unknown=True):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        if fixed_silence_unknown:
            skip = 2
            fixed_class = torch.tensor([SILENCE_INDEX, UNKNOWN_WORD_INDEX])
            n_way = n_way-skip
            self.sampling = []
            for i in range(self.n_episodes): 
                selected = torch.randperm(self.n_classes - skip)[:n_way]
                selected = torch.cat((fixed_class, selected.add(skip)))
                self.sampling.append(selected)        
        else:
            self.sampling = [torch.randperm(self.n_classes)[:self.n_way] for i in range(self.n_episodes)]

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield self.sampling[i]
            

def prepare_words_list(wanted_words, silence, unknown):
    extra_words = []
    if silence:
        extra_words.append(SILENCE_LABEL)
    if unknown:
        extra_words.append(UNKNOWN_WORD_LABEL)
    return extra_words + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
    # Split dataset in training, validation, and testing set
    # Should be modified to load validation data from validation_list.txt
    # Should be modified to load testing data from testing_list.txt

    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result




class GSCSpeechDataset:
    def __init__(self, data_dir, GSCtype, cuda, args):
        self.sample_rate = args['sample_rate']
        self.clip_duration_ms = args['clip_duration'] 
        self.window_size_ms = args['window_size']
        self.window_stride_ms = args['window_stride']
        self.n_mfcc = args['n_mfcc']
        self.feature_bin_count = args['num_features']
        self.foreground_volume = args['foreground_volume']
        self.time_shift_ms = args['time_shift']
        self.desired_samples = int(self.sample_rate * self.clip_duration_ms / 1000)

        # by now silence and background are enabled by default        
        self.use_background = args['include_noise']
        self.background_volume = args['bg_volume']
        self.background_frequency= args['bg_frequency']
        
        self.silence = args['include_silence']
        self.silence_num_samples = args['num_silence']
        self.unknown = args['include_unknown']
        
        
        self.data_cache = {}
        self.data_dir = data_dir
        
        # this are properties of the dataset!
        GSC_training_parameters = {
            'silence_percentage':10.0,
            'unknown_percentage':10.0,
            'validation_percentage':10.0,
            'testing_percentage':10.0,
        }
        unknown_words = ['backward','forward','visual','follow','learn']
        if GSCtype == 'GSC12':
            #self.silence = True
            target_words='yes,no,up,down,left,right,on,off,stop,go,'  # GSCv2 - 12 words
            print('10 word')
        elif GSCtype == 'GSC22':
            self.silence = True
            target_words='bed,bird,cat,dog,eight,five,four,happy,house,marvin,nine,one,seven,sheila,six,three,tree,two,wow,zero,'  # GSCv2 - 12 words
            unknown_words = []
            print('20 word')
            
        elif GSCtype == 'GSC10': #meta train task
            target_words='bed,bird,cat,dog,eight,five,four,nine,one,seven,six,three,tree,two,zero,'
            unknown_words = []
            print('10 words for meta train taks')
        elif GSCtype == 'GSC5': #meta val task
            target_words='happy,house,marvin,sheila,wow,'
            unknown_words = []
            print('5 words for meta val taks')
        else:
            # Selecting 35 words
            print('35 word - 5 words')
            target_words='yes,no,up,down,left,right,on,off,stop,go,bed,bird,cat,dog,eight,five,four,happy,house,marvin,nine,one,seven,sheila,six,three,tree,two,wow,zero,'  # GSCv2 - 35 words
        wanted_words=(target_words).split(',')
        wanted_words.pop()
        GSC_training_parameters['wanted_words'] = wanted_words
        GSC_training_parameters['unknown_words'] = unknown_words


        self.generate_data_dictionary(GSC_training_parameters)
        
        self.background_data = self.load_background_data()
        
        #try if can I include cuda here
        self.cuda = cuda
        self.max_class = len(wanted_words)

#        self.mfcc = self.build_mfcc_extractor()
#        if cuda:
#            self.mfcc.cuda()

    def get_episodic_fixed_sampler(self, num_classes,  n_way, n_episodes, fixed_silence_unknown = False, include_unknown = True):
        return EpisodicFixedBatchSampler(num_classes, n_way, n_episodes, fixed_silence_unknown = fixed_silence_unknown, include_unknown=include_unknown)
                    
    def get_episodic_dataloader(self, set_index, n_way, n_samples, n_episodes, sampler='episodic', 
            include_silence=True, include_unknown=True, unique_speaker=False):

        #if cuda:
         #   self.transforms.append(CudaTransform())
        #transforms = compose(self.transforms)

        # exclude silence and unknown from the list
        class_list = []
        for item in self.words_list:
            if not include_silence and item == SILENCE_LABEL:
                continue
            if not include_unknown and item == UNKNOWN_WORD_LABEL:
                continue
            class_list.append(item)
        
        if sampler == 'episodic':
            sampler = self.get_episodic_fixed_sampler(len(class_list),  
                            n_way, n_episodes)

        dl_list=[]        
        if set_index in ['training', 'testing']:
            for keyword in class_list:
                ts_ds = self.get_transform_dataset(self.data_set[set_index], [keyword])
                
                if n_samples <= 0:
                    n_support = len(ts_ds)
                
                dl = torch.utils.data.DataLoader(ts_ds, batch_size=n_samples, 
                        shuffle=True, num_workers=0)
                dl_list.append(dl)

            ds = SetDataset(dl_list)

            data_loader_params = dict(batch_sampler = sampler,  num_workers =8, 
                    pin_memory=not self.cuda)   
            dl = torch.utils.data.DataLoader(ds, **data_loader_params)
        else:
            raise ValueError("Set index = {} in episodic dataset is not correct.".format(set_index))

        return dl
    

    def get_iid_dataloader(self, set_index, batch_size, class_list = False, include_silence=True, include_unknown=True, unique_speaker=False):
        
        # exclude silence and unknown from the list
        if not class_list:
            class_list = []
            for item in self.words_list:
                if not include_silence and item == SILENCE_LABEL:
                    continue
                if not include_unknown and item == UNKNOWN_WORD_LABEL:
                    continue
                class_list.append(item)
            
        ts_ds = self.get_transform_dataset(self.data_set[set_index], class_list)
        dl = torch.utils.data.DataLoader(ts_ds, batch_size=batch_size, shuffle=True, num_workers=0)

        return dl

    def dataset_filter_class(self, dslist, classes):
    # FIXME: by now unique_speaker are not handled
        filtered_ds = []
        for item in dslist:
            label = item['label']
            if label in classes:
                filtered_ds.append(item)
            
        return filtered_ds
    
    def get_transform_dataset(self, file_dict, classes, filters=None):
        # file dict include is [{ 'label': LABEL_str, 'file': file_path, 'speaker': spkr_id}, .. ]
        # classes is a list of classes
        transforms = compose([
                partial(self.load_audio, 'file', 'label', 'data'),
                partial(self.adjust_volume, 'data'),
                partial(self.shift_and_pad, 'data'),
                partial(self.mix_background, self.use_background,'data', 'label'),
#                partial(self.extract_features, 'data', 'feat'),
                partial(self.label_to_idx, 'label', 'label_idx')

        ])
        file_dict = self.dataset_filter_class(file_dict, classes)
        ls_ds = ListDataset(file_dict)
        ts_ds = TransformDataset(ls_ds, transforms)
        
        return ts_ds
    
    def num_classes(self):
        return len(self.words_list)
        
    def label_to_idx(self, k, key_out, d):
        label_index = self.word_to_index[d[k]]
        d[key_out] = torch.LongTensor([label_index]).squeeze()
        return d

    def mix_background(self, use_background, k, key_label, d):
        foreground = d[k]
        if use_background or d[key_label] == SILENCE_LABEL: # add background noise for silence samples
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
                bg_vol = np.random.uniform(0, self.background_volume)
            else:
                bg_vol = 0
        else:
            background_reshaped = torch.zeros(1, self.desired_samples)
            bg_vol = 0

        background_mul = background_reshaped * bg_vol
        background_add = background_mul + foreground
        background_clamped = torch.clamp(background_add, -1.0, 1.0)
        d[k] = background_clamped
        return d
    
    def extract_features(self, k, key_out, d):
        
        if self.cuda:
            d_in = d[k].cuda()
        else:
            d_in = d[k]
        features = self.mfcc(d_in)[0] # just one channel
        features = features[:self.feature_bin_count]
        features = features.T # f x t -> t x f
        d[key_out] = torch.unsqueeze(features,0)
        return d

    def load_background_data(self):
        background_path = os.path.join(self.data_dir, '_background_noise_', '*.wav')
        background_data = []
        if self.use_background or self.silence:
            for wav_path in glob.glob(background_path):
                bg_sound, bg_sr = torchaudio.load(wav_path)
                background_data.append(bg_sound.flatten())
        return background_data
    
    def build_mfcc_extractor(self):

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

    def shift_and_pad(self, key, d):
        audio = d[key]
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

    
    def adjust_volume(self, key, d):
        d[key] =  d[key] * self.foreground_volume
        return d
    
    def load_audio(self, key_path, key_label, out_field, d):
        #{'label': 'stop', 'file': '../../data/speech_commands/GSC/stop/879a2b38_nohash_3.wav', 'speaker': '879a2b38'}
        sound, _ = torchaudio.load(filepath=d[key_path], normalize=True,
                                         num_frames=self.desired_samples)
        # For silence samples, remove any sound
        if d[key_label] == SILENCE_LABEL:
             sound.zero_()
        d[out_field] = sound
        return d

    def generate_data_dictionary(self, training_parameters):
        # For each data set, generate a dictionary containing the path to each file, its label, and its speaker.
        # Make sure the shuffling and picking of unknowns is deterministic.
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        wanted_words_index = {}
        unknown_words = training_parameters['unknown_words']
        
        global SILENCE_INDEX
        skip = 0
        if self.silence:
            skip +=1
        if self.unknown:
            skip +=1
        else:
            SILENCE_INDEX = SILENCE_INDEX -1
        
        for index, wanted_word in enumerate(training_parameters['wanted_words']):
            wanted_words_index[wanted_word] = index + skip

        # Prepare data sets
        self.data_set = {'validation': [], 'testing': [], 'training': []}
        unknown_set = {'validation': [], 'testing': [], 'training': []}
        all_words = {}
        # Find all audio samples
        search_path = os.path.join(self.data_dir, '*', '*.wav')
        
        #parse the folders
        for wav_path in glob.glob(search_path):
            _ , word = os.path.split(os.path.dirname(wav_path))
            split_wav_path = wav_path.split('/')
            ind = len(split_wav_path) - 1
            speaker_id = split_wav_path[ind].split('_')[0]  # Hardcoded, should use regex.
            word = word.lower()

            # Ignore background noise, as it has been handled by generate_background_noise()
            if word == BACKGROUND_NOISE_LABEL:
                continue

            all_words[word] = True
            # Determine the set to which the word should belong
            set_index = which_set(wav_path, training_parameters['validation_percentage'], training_parameters['testing_percentage'])

            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            # If we use 35 classes - all are known, hence no unkown samples      
            if word in wanted_words_index:
                self.data_set[set_index].append({'label': word, 'file': wav_path, 'speaker': speaker_id})
            elif word in unknown_words:
                unknown_set[set_index].append({'label': UNKNOWN_WORD_LABEL, 'file': wav_path, 'speaker': speaker_id})

        # store dictionary of words
        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        
        for index, wanted_word in enumerate(training_parameters['wanted_words']):
            if wanted_word not in all_words:
                raise Exception('Expected to find ' + wanted_word +
                                ' in labels but only found ' +
                                ', '.join(all_words.keys()))


        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_set['training'][0]['file']


        # Add silence and unknown words to each set
        for set_index in ['validation', 'testing', 'training']:

            set_size = len(self.data_set[set_index])
            if self.silence:
                silence_size = int(math.ceil(set_size * training_parameters['silence_percentage'] / 100))
                for _ in range(silence_size):
                    self.data_set[set_index].append({
                        'label': SILENCE_LABEL,
                        'file': silence_wav_path,
                        'speaker': "None" 
                    })
            
            if self.unknown:
                # Pick some unknowns to add to each partition of the data set.
                rand_unknown = random.Random(RANDOM_SEED)
                rand_unknown.shuffle(unknown_set[set_index])
                unknown_size = int(math.ceil(set_size * training_parameters['unknown_percentage'] / 100))
                self.data_set[set_index].extend(unknown_set[set_index][:unknown_size])

        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            rand_data_order = random.Random(RANDOM_SEED)
            rand_data_order.shuffle(self.data_set[set_index])

        # Prepare the rest of the result data structure.
        self.words_list = prepare_words_list(training_parameters['wanted_words'], self.silence, self.unknown)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
#            elif word in unknown_words:
#                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        if self.silence:
            self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX
        if self.unknown:
            self.word_to_index[UNKNOWN_WORD_LABEL] = UNKNOWN_WORD_INDEX
        
        # restor randomess
        #t = int(1000 * time.time())  % 2**32 # current time in milliseconds
        #random.seed(t)
        #np.random.seed(t)







    
