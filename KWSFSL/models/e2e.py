import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from models.utils import register_model
from models.encoder.DSCNN import DSCNNL, DSCNNM, DSCNNS, DSCNNL_NONORM, DSCNNL_LAYERNORM
from models.encoder.resnet import Res15, Res8
from models.encoder.TCResNet import TCResNet8, TCResNet8Dilated

from models.preprocessing import MFCC


from .utils import euclidean_dist

class E2Enet(nn.Module):
    def __init__(self, encoder, x_dim, preprocessing, num_classes):
        super(E2Enet, self).__init__()
        self.encoder = encoder

        self.criterion = nn.CrossEntropyLoss()
        self.preprocessing = preprocessing
        x_fake = torch.Tensor(1,x_dim[0],x_dim[1],x_dim[2] )
        z = self.encoder.forward(x_fake)
        self.classifier = nn.Linear(z.size(1), num_classes)

    def forward(self, x ):
        z = self.get_embeddings(x)
        y = self.classifier(z)
        return y

    def get_embeddings(self, x):
        # x is a batch of data
        if self.preprocessing:
            x = self.preprocessing.extract_features(x)
        zq = self.encoder.forward(x)
        return zq
        
    def loss_class(self, x, labels):
        
        y = self.forward(x)
        
        outputs = F.softmax(y, dim=1)
        loss = self.criterion(outputs, labels)
        
        # Compute training statistics
        scores, predicted = torch.max(outputs.data, 1)
        correct = torch.eq(predicted, labels).float()
        acc_val = correct.mean()
        
        score_mean = scores.mean()
        score_corr = scores.mul(correct).mean()
        score_wrong = scores.mul(correct.mul(-1).add(1)).mean()
        return loss, {
            'p_y': outputs,
            'loss': loss.item(),
            'acc': acc_val.item(),
            'score_mean': score_mean.item(),
            'score_corr': score_corr.item(),
            'score_wrong': score_wrong.item()
        }


def get_encoder(encoding, x_dim, hid_dim, out_dim):
    if encoding == 'DSCNNL':
        return DSCNNL(x_dim)
    elif encoding == 'DSCNNL_NONORM':
        return DSCNNL_NONORM(x_dim)
    elif encoding == 'DSCNNL_LAYERNORM':
        return DSCNNL_LAYERNORM(x_dim)        
    elif encoding == 'DSCNNM':
        return DSCNNM(x_dim)
    elif encoding == 'DSCNNS':
        return DSCNNS(x_dim)
    elif encoding == 'Resnet15':
        return Res15(hid_dim)
    elif encoding == 'Resnet8':
        return Res8(hid_dim)
    elif encoding == 'TCResNet8':
        return TCResNet8(x_dim[0], x_dim[1], x_dim[2])
    elif encoding == 'TCResNet8Dilated':
        return TCResNet8Dilated(x_dim[0], x_dim[1], x_dim[2])
    else:
        raise ValueError("Model {} is not valid".format(encoding))


@register_model('e2e_conv')
def load_e2e_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    encoding = kwargs['encoding']
    num_classes = kwargs['num_classes']

    # get encoder
    encoder = get_encoder(encoding, x_dim, hid_dim, z_dim)
    
    # get preprocessing
    preprocessing = False
    if 'mfcc' in kwargs.keys():
        audio_prep = kwargs['mfcc']
        preprocessing = MFCC(audio_prep)

    return E2Enet(encoder, x_dim, preprocessing, num_classes)
