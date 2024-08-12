
import os
import glob
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np


def get_default_augm_args():
    return {'type': 'noise',
             'data_dir': '', # datapath of the noise files
             'files_num':5,
             'wav_len':16000,
             'wav_sr':16000,
             'snr':0}


class OnDeviceAugment:
    def __init__(self, aug_arg):
        self.type = aug_arg['type']

        if self.type == 'noise':
            self.data_dir = aug_arg['data_dir']
            self.file_ext = '*.wav'
            self.files_num = aug_arg['files_num']
            self.wav_len = aug_arg['wav_len']
            self.wav_sr = aug_arg['wav_sr']
            self.snr = aug_arg['snr']

            
            noise_path = os.path.join(self.data_dir, self.file_ext )
            self.background_data = []
            available_noises = glob.glob(noise_path)
            n_noise = len(available_noises)
            assert self.files_num < n_noise, "Not enough noise availables"
            index = torch.randperm(n_noise)[:self.files_num].tolist()
            for wav_idx in index:
                wav_path = available_noises[wav_idx]
                bg_sound, bg_sr = torchaudio.load(wav_path)
                if bg_sr != self.wav_sr:
                    bg_sound = F.resample(bg_sound, bg_sr, self.wav_sr)

                bg_sound = bg_sound.flatten()
                background_offset = np.random.randint(
                    0, len(bg_sound) - self.wav_len)
                background_clipped = bg_sound[background_offset:(
                    background_offset + self.wav_len)]
                background_reshaped = background_clipped.reshape([1, self.wav_len])

                self.background_data.append(background_reshaped)
        else:
            raise ValueError("On Device Aug Option {} is not valid".format(self.type))

    def apply_batch(self, x):
        batch_sz = x.size(0)
        print('Size of input data: ', x.size())
        out_data = []
        for i in range(batch_sz):
            x_o = self.apply(x[i]).unsqueeze(0)
            out_data.append(x_o)
        
        out_data = torch.cat(out_data, dim=0)
        print('Size of output data: ', out_data.size())
        return out_data

        
    def apply(self, x):
        if self.type == 'noise':
            noise_idx = np.random.randint(self.files_num)
            background_reshaped = self.background_data[noise_idx]
            s_pow = x.pow(2).sum()
            n_pow = background_reshaped.pow(2).sum()
            bg_vol = (s_pow/((10**self.snr/10)*n_pow)).sqrt().item()

            background_mul = background_reshaped * bg_vol
            if x.is_cuda:
                background_mul = background_mul.cuda()
            background_add = background_mul + x
            background_clamped = torch.clamp(background_add, -1.0, 1.0)
            return background_clamped
            
        else:
            raise ValueError("Apply of Option {} not supported".format(self.type))


                    
    
if __name__ == '__main__':
    aug_arg={'type': 'noise',
             'data_dir': '', # FIXME: datapath of the noise files
             'files_num':5,
             'wav_len':16000,
             'wav_sr':16000,
             'snr':0}
    OnDeviceAugment(aug_arg)
