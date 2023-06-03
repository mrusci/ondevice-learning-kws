import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from models.utils import register_model
from models.encoder.DSCNN import DSCNNL, DSCNNM, DSCNNS, DSCNNL_NONORM, DSCNNL_LAYERNORM, DSCNNS_NONORM, DSCNNS_LAYERNORM
from models.encoder.resnet import Res15, Res8
from models.encoder.TCResNet import TCResNet8, TCResNet8Dilated

from models.preprocessing import MFCC

from models.losses.triplet import online_triplet_loss
from models.losses.protonet import prototypical_loss
from models.losses.angproto import angular_proto_loss
#from models.losses.normsoftmax import norm_softmax
from models.losses.amsoftmax import am_softmax

# comparison with peeler
from models.encoder.DSCNN import DSCNNS_PEELER, DSCNNL_PEELER
from models.losses.peeler import peeler_loss
from models.losses.dproto import dproto



class ReprModel(nn.Module):
    def __init__(self, encoder, preprocessing, 
            criterion, x_dim, emb_norm, feat_extractor=False):
        super(ReprModel, self).__init__()
        self.encoder = encoder
        self.preprocessing = preprocessing
        self.emb_norm = emb_norm

        # get embedding size
        x_fake = torch.Tensor(1,x_dim[0],x_dim[1],x_dim[2] )
        z = self.encoder.forward(x_fake)
        z_dim = z.size(1)

        #setup loss
        if criterion['type'] == 'prototypical':
            self.criterion = prototypical_loss(criterion)
        elif criterion['type'] == 'triplet':
            self.criterion = online_triplet_loss(criterion)
        elif criterion['type'] == 'angproto':
            self.criterion = angular_proto_loss(criterion)
        elif criterion['type'] == 'normsoftmax':
            criterion['z_dim'] = z_dim
            criterion['margin'] = 0
            self.criterion = am_softmax(criterion, scale=1)
        elif criterion['type'] == 'amsoftmax':
            criterion['z_dim'] = z_dim
            self.criterion = am_softmax(criterion)
        elif criterion['type'] == 'peeler':
            criterion['z_dim'] = z_dim
            self.criterion = peeler_loss(criterion)
        elif criterion['type'] == 'dproto':
            criterion['z_dim'] = z_dim
            self.criterion = dproto(criterion)    

        self.feat_extractor = feat_extractor
    
    def get_embeddings(self, x):
        # x is a batch of data
        if self.preprocessing:
            x = self.preprocessing.extract_features(x)
        if self.feat_extractor:
            zq = zq 
        zq = self.encoder.forward(x)
        if self.emb_norm:
            zq = F.normalize(zq, p=2.0, dim=-1)
        return zq

    def loss(self, x):
        # get information
        n_class = x.size(0)
        n_sample = x.size(1)

        #  inference
        x = x.view(n_class * n_sample, *x.size()[2:]).cuda()
        zq = self.get_embeddings(x)
        
        # loss
        loss_val = self.criterion.compute(zq, n_sample, n_class)

        return loss_val, {
            'loss': loss_val.item(),
        }

    def loss_class(self, x, labels):
        zq = self.get_embeddings(x)
        return self.criterion.compute(zq, labels)



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
    elif encoding == 'DSCNNS_NONORM':
        return DSCNNS_NONORM(x_dim)
    elif encoding == 'DSCNNS_LAYERNORM':
        return DSCNNS_LAYERNORM(x_dim)    
    
    # experiments for PEELER
    elif encoding == 'DSCNNS_PEELER':
        return DSCNNS_PEELER(x_dim)
    elif encoding == 'DSCNNL_PEELER':
        return DSCNNL_PEELER(x_dim)    
    

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


@register_model('repr_conv')
def load_repr_conv(**kwargs):
    z_norm = kwargs['z_norm']
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    encoding = kwargs['encoding']
    print(encoding, x_dim, hid_dim, z_dim)

    #get encoder
    encoder = get_encoder(encoding, x_dim, hid_dim, z_dim)

    # get preprocessing
    preprocessing = False
    if 'mfcc' in kwargs.keys():
        audio_prep = kwargs['mfcc']
        preprocessing = MFCC(audio_prep)

    #get criterion 
    criterion = kwargs['loss'] if 'loss' in kwargs.keys() else False

    # get feat_extractor stage, e.g. wav2vec
    feat_extractor = False

    return ReprModel(encoder, preprocessing, criterion, x_dim, z_norm, feat_extractor)
