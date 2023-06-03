# adapted from https://github.com/tyler-hayes/Embedded-CL

import os
import torch
from torch import nn
import torch.nn.functional as F

from models.utils import euclidean_dist
from models.losses.dproto import dproto


class DProto(nn.Module):
    """
    This is an implementation of the Nearest Class Mean algorithm for streaming learning.
    """

    def __init__(self, backbone=None, cuda=True):
        """
        Init function for the NCM model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        """

        super(DProto, self).__init__()

        # NCM parameters
        self.cuda = cuda
        self.input_shape = None
        self.num_classes = None
        self.class_list = None
        self.word_to_index = {}

        # feature extraction backbone
        self.backbone = backbone
        assert isinstance(backbone.criterion, deepset)or isinstance(backbone.criterion, dsfeatproto), 'Model not trained with DeepSet loss'

        if backbone is not None:
            self.backbone = backbone.eval()
            if cuda:    
                self.backbone.cuda()
                if hasattr(self.backbone.preprocessing, 'mfcc'):
                    self.backbone.preprocessing.mfcc.cuda()

        self.proto = None
        self.proto_u = None 


    @torch.no_grad()
    def predict(self, X, return_probas=False):
        
        n_samples =  X.size(0)
        scores = - euclidean_dist(X, self.proto) / self.backbone.criterion.temp_knw

        scores_u = - euclidean_dist(X, self.proto_u) / self.backbone.criterion.temp_unk
        max_score_u =  scores_u.max(1)[0].unsqueeze(1)
        scores = torch.cat([scores, max_score_u],dim=1)
        print(X, scores, scores_u, max_score_u.size())

        p_y = torch.softmax(scores, dim=1)

        # return predictions or probabilities
        if not return_probas:
            return scores.cpu()
        else:
            return p_y.cpu()

    @torch.no_grad()
    def fit_batch_offline(self, x, class_list):
        """
        Fit the NCM model to a new sample (x,y).
        :param item_ix:
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        if self.cuda:
            x = x.cuda()
            


        # class list and define class2idb
        # append the unknown class at the end
        self.class_list = class_list
        j = 0
        for item in class_list:
            if item == '_unknown_':
                unk_idx = j
            else:
                self.word_to_index[item] = j
                j += 1
        self.word_to_index['_unknown_'] = len(class_list)-1

        # remove unknown from the support set
        x_idx = [j for j in range(x.size(0)) if j != unk_idx]
        print(unk_idx, x_idx)
        x = x[x_idx,:,:,:]


        # compute vectors 
        self.num_classes = x.size(0)
        n_support = x.size(1)
        self.input_shape = x.size(2)

        # compute vectors for snatcherf
        x = x.view(self.num_classes * n_support, *x.size()[2:])
        zq = self.backbone.get_embeddings(x)
        zq = zq.view(self.num_classes, n_support, *zq.size()[1:])
        support_samples = zq.contiguous()
        
        # compute protoptypes using support vectors
        proto_samples = support_samples.mean(1)
        self.proto = proto_samples
        self.proto_u = self.backbone.criterion.set_func(proto_samples) 

        print('proto:', self.proto.size())
        for i in range(self.proto.size(0)):
            print(self.proto[i])
        print('proto u:', self.proto_u.size())
        for i in range(self.proto_u.size(0)):
            print(self.proto_u[i])
        #exit(0)





    def class2torchidx(self, labels):
        label_list = []
        for item in labels:
            label_index = self.word_to_index[item]
            label_list.append([label_index])
        return torch.LongTensor(label_list)

    @torch.no_grad()
    def evaluate_batch(self, test_x, labels, return_probas=False):
        # test_x and labels are batches of data

        if isinstance(labels[0], str):
            target_inds = self.class2torchidx(labels)

        if self.cuda:
            test_x = test_x.cuda()
            target_inds = target_inds.cuda()

        if self.backbone is not None:
            zq = self.backbone.get_embeddings(test_x)
        else:
            zq = test_x

        p_y = self.predict(zq, return_probas=True)
        ##zeros_score = torch.zeros(p_y.size(0), 1)
        ##p_y = torch.cat([p_y,zeros_score],dim=1)

        return p_y, target_inds
