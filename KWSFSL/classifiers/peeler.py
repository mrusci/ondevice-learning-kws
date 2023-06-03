# adapted from https://github.com/tyler-hayes/Embedded-CL

import os
import torch
from torch import nn
import torch.nn.functional as F

from models.utils import euclidean_dist
from models.losses.peeler import peeler_loss


class PeelerClass(nn.Module):
    """
    This is an implementation of the Nearest Class Mean algorithm for streaming learning.
    """

    def __init__(self, backbone=None, cuda=True):
        """
        Init function for the NCM model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        """

        super(PeelerClass, self).__init__()

        # NCM parameters
        self.cuda = cuda
        self.input_shape = None
        self.num_classes = None
        self.class_list = None
        self.word_to_index = {}

        # feature extraction backbone
        self.backbone = backbone
        assert isinstance(backbone.criterion, peeler_loss), 'Model not trained with PEELER loss'
        print(self.backbone)

        params=0
        for p in list(self.backbone.criterion.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            params += nn
        print('number of parameters:', params)
        exit(0)
        if backbone is not None:
            self.backbone = backbone.eval()
            if cuda:    
                self.backbone.cuda()
                if hasattr(self.backbone.preprocessing, 'mfcc'):
                    self.backbone.preprocessing.mfcc.cuda()

        self.sigs = None
        self.sigs = None


    @torch.no_grad()
    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """        
        # inference
        batch_size, feat_size, feat_h, feat_w = X.size()
        query_mu_whitten = torch.mul(X.unsqueeze(1), self.sigs.unsqueeze(0))

        query_mu_whitten = self.backbone.criterion.avgpool(query_mu_whitten.view(-1, feat_size, feat_h, feat_w))
        query_mu_whitten = query_mu_whitten.view(batch_size, -1, feat_size)
#        print(query_mu_whitten.size(), self.mu_whitten.size())
        dist_few = torch.norm(query_mu_whitten - self.mu_whitten.unsqueeze(0), p=2, dim=2)
#        print(dist_few.size())
        scores = -dist_few 
        if not return_probas:
            return scores.cpu()
        
        p_y = F.softmax(scores, dim=1).cpu()
        # add fictitious scor 0.0 for the _unknown_class
        zeros_score = torch.zeros(p_y.size(0), 1)
        p_y = torch.cat([p_y,zeros_score],dim=1)
        return p_y

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



        self.num_classes = x.size(0)
        n_support = x.size(1)
        self.input_shape = x.size(2)

        # compute vectors for peeler
        x = x.view(self.num_classes * n_support, *x.size()[2:])
        support_mu = self.backbone.get_embeddings(x)

        support_sigs_0 = support_mu
        support_sigs_1 = self.backbone.criterion.layer_sigs_0(support_sigs_0).mean(dim=0, keepdim=True).expand_as(support_sigs_0)
        support_sigs_1 = torch.cat((support_sigs_0, support_sigs_1), dim=1)
        support_sigs = self.backbone.criterion.layer_sigs_1(support_sigs_1)

        batch_size, feat_size, feat_h, feat_w = support_mu.size()

        # fewshot
        mu = support_mu.view(self.num_classes, n_support, feat_size, feat_h, feat_w).mean(dim=1)
        self.sigs = support_sigs.view(self.num_classes, n_support, feat_size, feat_h, feat_w).mean(dim=1)

        mu_whitten = torch.mul(mu, self.sigs)
        self.mu_whitten = self.backbone.criterion.avgpool(mu_whitten).view(-1, feat_size)
        

        print('signs:', self.sigs.size())
        print('mu_whitten:', self.mu_whitten.size())



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

        return p_y, target_inds
