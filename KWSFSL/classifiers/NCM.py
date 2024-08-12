# adapted from https://github.com/tyler-hayes/Embedded-CL

import os
import torch
from torch import nn
import torch.nn.functional as F

from models.utils import euclidean_dist
from classifiers.augment import OnDeviceAugment, get_default_augm_args

class NearestClassMean(nn.Module):
    """
    This is an implementation of the Nearest Class Mean algorithm for streaming learning.
    """

    def __init__(self, backbone=None, cuda=True):
        """
        Init function for the NCM model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        """

        super(NearestClassMean, self).__init__()

        # NCM parameters
        self.cuda = cuda
        self.input_shape = None
        self.num_classes = None
        self.class_list = None
        self.word_to_index = {}

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval()
            self.backbone.encoder.return_feat_maps = False

            if cuda:    
                self.backbone.cuda()
                if hasattr(self.backbone.preprocessing, 'mfcc'):
                    self.backbone.preprocessing.mfcc.cuda()
            if hasattr(self.backbone.criterion, 'distance'):
                print('Used Distance: ', self.backbone.criterion.distance)
            else:
                self.backbone.criterion.distance = 'cosine' # euclidean or cosine
            if self.backbone.criterion.distance == 'cosine':
                self.backbone.emb_norm = True
                self.backbone.criterion.distance = 'euclidean'


        # setup weights for NCM. not initialized yet.
        self.muK = None
        self.cK = None
        self.num_updates = 0


    @torch.no_grad()
    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        if hasattr(self.backbone.criterion, 'distance'):
            if self.backbone.criterion.distance == 'euclidean': 
                scores = - euclidean_dist(X, self.muK)
            elif self.backbone.criterion.distance == 'cosine': 
                #print(X.size(), self.muK.size())
                scores = F.cosine_similarity( 
                    X.unsqueeze(1).expand( X.size(0), *self.muK.size()), self.muK, dim=2)
                #print(scores.size())

            else:
                raise ValueError('Type of distance metric {} not supported'.format(
                    self.backbone.criterion.distance))
        else:
            scores = - euclidean_dist(X, self.muK)

        # return predictions or probabilities
        if not return_probas:
            return scores.cpu()
        else:
            return torch.softmax(scores, dim=1).cpu()


    @torch.no_grad()
    def fit_batch_offline(self, x, class_list, online_update=False, augument_proto=0, arg_augm=None):
        """
        Fit the NCM model to a batch of data
        - x is a tensor of size (n_classes, n_support, wav_size)
        - class_list is a string list of classes ordered as x
                len(class_list) == x.size(0)
        - online_update, if true running mean is used
        - augument_proto, if > 1 every prototype is augmented with augument_proto-1 distorted samples (default: 0)
        - augment_type, describes the type of augumentation (default: None)
        """
        if self.cuda:
            x = x.cuda()
            
        self.num_classes = x.size(0)
        n_support = x.size(1)
        self.input_shape = x.size(2)

        # class list and define class2idb
        assert self.num_classes == len(class_list), "Mismatch between class list and x size"
        self.class_list = class_list
        for i,item in enumerate(class_list):
            self.word_to_index[item] = i

        if augument_proto > 1:
            if arg_augm is None:
                arg_augm = get_default_augm_args()
            self.augm = OnDeviceAugment(arg_augm)
        else:
            augument_proto = 1

        # inference
        if online_update:
            if self.cK is None:
                self.cK = torch.zeros(self.num_classes)

            for c in range(self.num_classes):
                for i in range(n_support):
                    for ag in range(augument_proto):
                        ns_c = self.cK[c] + 1 
                        x_c_i =  x[c,i,:]
                        if ag>1:
                            x_c_i = self.augm.apply(x_c_i)
                        zq = self.backbone.get_embeddings(x_c_i.unsqueeze(0))

                        if self.muK is None:
                            emb_sz = zq.size(-1)
                            self.muK = torch.zeros(self.num_classes, emb_sz)
                            if self.cuda:
                                self.muK = self.muK.cuda()
                                
                        if ns_c == 1:
                            self.muK[c] = zq[0]
                        else:
                            self.muK[c] = (self.muK[c].mul(ns_c-1) + zq[0]) / ns_c

                        self.cK[c] = ns_c

        else:
            x = x.view(self.num_classes * n_support, *x.size()[2:])
            zq = self.backbone.get_embeddings(x)
            z_proto = zq.view(self.num_classes, n_support, zq.size(-1)).mean(1)

            # update class means
            self.muK = z_proto
            self.cK = torch.ones(self.num_classes).mul(n_support)
            self.num_updates = 0
            
        print('Classfier fit!')


    def class2torchidx(self, labels):
        label_list = []
        for item in labels:
            label_index = self.word_to_index[item]
            label_list.append([label_index])
        return torch.LongTensor(label_list)

    @torch.no_grad()
    def evaluate_batch(self, test_x, labels, return_probas=False):
        if isinstance(labels[0], str):
            target_inds = self.class2torchidx(labels)

        if self.cuda:
            test_x = test_x.cuda()
            target_inds = target_inds.cuda()

        if self.backbone is not None:
            zq = self.backbone.get_embeddings(test_x)
        else:
            zq = test_x

        scores = self.predict(zq, return_probas=return_probas)

        if return_probas:
            scores = F.softmax(scores, dim=1).cpu()
                    
        return scores, target_inds
        
