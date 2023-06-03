import argparse

parser = argparse.ArgumentParser(description='Train feature extractor')

#default values
default_model_name = 'e2e_conv'
default_encoding = 'DSCNNL'
default_model_prep = 'mfcc'

default_dataset = 'googlespeechcommand'
default_datadir = '' # FIXME: add here your path
default_split = 'GSC12'
# model args
parser.add_argument('--model.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="model name (default: {:s})".format(default_model_name))
parser.add_argument('--model.x_dim', type=str, default='1,49,10', metavar='XDIM',
                    help="dimensionality of input images (default: '1,28,28')")
parser.add_argument('--model.hid_dim', type=int, default=64, metavar='HIDDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--model.z_dim', type=int, default=64, metavar='ZDIM',
                    help="dimensionality of input images (default: 64)")
parser.add_argument('--model.encoding', type=str, default=default_encoding, metavar='MODELENC',
                    help="model encoding (default: {:s})".format(default_encoding))
parser.add_argument('--model.model_path', type=str, default="", metavar='MODELPATH',
                    help="location of pretrained model to evaluate (default: empty)")
parser.add_argument('--model.z_norm', action='store_true', 
                    help="normalize feature vector (default: False)")
parser.add_argument('--model.preprocessing', type=str, default=default_model_prep, metavar='MODELPREP',
                    help="model preprocessing (default: {:s})".format(default_model_prep))                   


# train args - used for train
parser.add_argument('--train.epochs', type=int, default=10000, metavar='NEPOCHS',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                    help='optimization method (default: Adam)')
parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.decay_every', type=int, default=20, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
parser.add_argument('--train.weight_decay', type=float, default=0.0, metavar='WD',
                    help="weight decay (default: 0.0)")
parser.add_argument('--train.patience', type=int, default=20, metavar='PATIENCE',
                    help='number of epochs to wait before validation improvement (default: 1000)')
parser.add_argument('--train.batch_size', type=int, default=128, metavar='BATCHSIZE',
                    help='batchsize when training with minibatch gradient descent (default: 128)')

# episodic train options - used for train
parser.add_argument('--train.n_episodes', type=int, default=200, 
                    help='train few shot learning: number of episodes  (default: 200)')
parser.add_argument('--train.n_way', type=int, default=12, 
                    help='train few shot learning: number of subclasses (default: 12)')
parser.add_argument('--train.n_support', type=int, default=0, 
                    help='train few shot learning: number of support samples (default: 5)')
parser.add_argument('--train.n_query', type=int, default=5, 
                    help='train few shot learning: number of query samples (default: 5)')
parser.add_argument('--train.n_way_u', type=int, default=0, 
                    help='train few shot learning: number of unknown subclasses (default: 0)')

# loss settings - used for train
parser.add_argument('--train.loss', type=str, default='metric',
                    help='FIXME')
parser.add_argument('--train.distance', type=str, default='euclidean',
                    help='FIXME')
parser.add_argument('--train.margin', type=float, default=0.5, 
                    help='FIXME')


# log args
default_fields = 'loss,acc'
parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                    help="fields to monitor during training (default: {:s})".format(default_fields))
default_exp_dir = 'results'
parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))
parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")


# speech data args
parser.add_argument('--speech.dataset', type=str, default=default_dataset, metavar='DS',
                    help="data set name (default: {:s})".format(default_dataset))
parser.add_argument('--speech.task', type=str, default=default_split, metavar='SP',
                    help="split name (default: {:s})".format(default_split))
parser.add_argument('--speech.default_datadir', type=str, default=default_datadir, metavar='DIR',
                    help="path to the dataset")
parser.add_argument('--speech.include_silence', action='store_true', help="one of the classes out of n should be silence (default: False)")
parser.add_argument('--speech.include_unknown', action='store_true', help="one of the classes out of n should be unknown (default: False)")
parser.add_argument('--speech.sample_rate', type=int, default=16000, help='desired sampling rate of the input')
parser.add_argument('--speech.clip_duration', type=int, default=1000, help='clip duration in milliseconds')
parser.add_argument('--speech.time_shift', type=int, default=100, help='time shift the audio in milliseconds')
parser.add_argument('--speech.bg_volume', type=float, default=0.1, help='background volumen to mix in between 0 and 1')
parser.add_argument('--speech.bg_frequency', type=float, default=1.0, help='Amount of samples that should be mixed with background noise (between 0 and 1)')
parser.add_argument('--speech.num_silence', type=int, default=1000, help='Number of silence samples to generate')
parser.add_argument('--speech.foreground_volume', type=float, default=1)

parser.add_argument('--speech.include_noise', action='store_true', help="one of the classes out of n should be unknown (default: False)")
parser.add_argument('--speech.noise_snr', type=int, default=5, help='time shift the audio in milliseconds')
parser.add_argument('--speech.noise_frequency', type=float, default=0.95, help='Amount of samples that should be mixed with background noise (between 0 and 1)')


# feature extraction
parser.add_argument('--speech.window_size', type=int, default=40)
parser.add_argument('--speech.window_stride', type=int,default=20)
parser.add_argument('--speech.num_features', type=int, default=10, help='Number of mfcc features to feed the model')
parser.add_argument('--speech.n_mfcc', type=int, default=40, help='Number of mfcc features to compute')

# classifier options - used for test
parser.add_argument('--fsl.classifier', type=str, default='ncm', 
                    help='Type of the classifier')
parser.add_argument('--fsl.test.n_way', type=int, default=12, 
                    help='test few shot learning: number of subclasses (default: 12)')
parser.add_argument('--fsl.test.n_support', type=int, default=5, 
                    help='test few shot learning: number of support samples (default: 5)')
parser.add_argument('--fsl.test.n_episodes', type=int, default=100, 
                    help='test few shot learning: number of episodes  (default: 100)')
parser.add_argument('--fsl.test.fixed_silence_unknown', action='store_true',
                    help='force unknown and silence class to be present in every episode (default: False)')
parser.add_argument('--fsl.test.batch_size', type=int, default=128, 
                    help='test few shot batch size  (default: 128)')
