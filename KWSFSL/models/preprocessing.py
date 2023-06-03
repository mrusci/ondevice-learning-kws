import torch
import torchaudio
import math 


class AudioPrep:
    def extract_features(self, x):
        return x

class MFCC(AudioPrep):
    
    def __init__(self, audio_prep):
        self.window_size_ms = audio_prep['window_size_ms']
        self.window_stride_ms = audio_prep['window_stride_ms']
        self.sample_rate = audio_prep['sample_rate']
        self.n_mfcc = audio_prep['n_mfcc']
        self.feature_bin_count = audio_prep['feature_bin_count']

        def next_power_of_2(x):  
            return 1 if x == 0 else 2**math.ceil(math.log2(x))
        
        frame_len = self.window_size_ms / 1000
        stride = self.window_stride_ms / 1000
        n_fft = next_power_of_2(frame_len*self.sample_rate)

        self.mfcc = torchaudio.transforms.MFCC(self.sample_rate,
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
        super(MFCC, self).__init__()

    def extract_features(self, x):
        
        with torch.no_grad():
            features = self.mfcc(x)
        features = torch.narrow(features, -2, 0, self.feature_bin_count)
        features = features.mT # f x t -> t x f

        return features



if __name__ == '__main__':
    audio_prep = {
        'window_size_ms': 40,
        'window_stride_ms': 20,
        'sample_rate': 16000,
        'n_mfcc': 40, 
        'feature_bin_count': 10,
    }
    mfcc = MFCC(audio_prep)
    x=torch.Tensor(5,16000)
    print(x.size())
    feat = mfcc.extract_features(x)
    print(feat.size())
