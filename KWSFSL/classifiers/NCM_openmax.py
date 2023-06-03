# adapted from https://github.com/tyler-hayes/Embedded-CL

import os
import torch
from torch import nn
import torch.nn.functional as F

import libmr
from models.utils import euclidean_dist

import glob 
import torchaudio
import numpy as np


class NCMOpenMax(nn.Module):
    """
    This is an implementation of the Nearest Class Mean algorithm for streaming learning.
    """

    def __init__(self, backbone=None, cuda=True):
        """
        Init function for the NCM model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        """

        super(NCMOpenMax, self).__init__()

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
            print(self.backbone)
            
            #model_parameters = self.backbone.encoder.parameters()
            #p1 = sum([np.prod(p.size()) for p in model_parameters])
            #model_parameters = self.backbone.criterion.parameters()
            #p2 = sum([np.prod(p.size()) for p in model_parameters])
            #print(p1,p2)

            #exit()
            if cuda:    
                self.backbone.cuda()
                if hasattr(self.backbone.preprocessing, 'mfcc'):
                    self.backbone.preprocessing.mfcc.cuda()
            if not hasattr(self.backbone.criterion, 'distance'):
                self.backbone.criterion.distance = 'cosine' # euclidean or cosine
            if self.backbone.criterion.distance == 'cosine':
                self.backbone.emb_norm = True

        # setup weights for NCM. not initialized yet.
        self.muK = None
        self.cK = None
        self.num_updates = 0

    @torch.no_grad()
    def find_dists(self, A, B):
        """
        Given a matrix of points A, return the indices of the closest points in A to B using L2 distance.
        :param A: N x d matrix of points
        :param B: M x d matrix of points for predictions
        :return: indices of closest points in A
        """
        M, d = B.shape
        with torch.no_grad():
            B = torch.reshape(B, (M, 1, d))  # reshaping for broadcasting
            square_sub = torch.mul(A - B, A - B)  # square all elements
            dist = torch.sum(square_sub, dim=2)
        return -dist

    @torch.no_grad()
    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """

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

        #print('D:' , scores)
        # mask off predictions for unseen classes
#        not_visited_ix = torch.where(self.cK == 0)[0]
#        min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
#        scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
#            len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes

        # return predictions or probabilities
        if not return_probas:
            return scores.cpu()
        else:
            return torch.softmax(scores, dim=1).cpu()

    def load_background_data(self):
        background_path = os.path.join('Datasets/GSC/', '_background_noise_', '*.wav')
        background_data = []
        for wav_path in glob.glob(background_path):
            bg_sound, bg_sr = torchaudio.load(wav_path)
            background_data.append(bg_sound.flatten())
        return background_data
    
    def get_background(self, len_foreground):
        background_index = np.random.randint(len(self.background_data))
        background_samples = self.background_data[background_index]
        if len(background_samples) <= len_foreground:
            raise ValueError(
                'Background sample is too short! Need more than %d'
                ' samples but only %d were found' %
                (len_foreground, len(background_samples)))
        background_offset = np.random.randint(
            0, len(background_samples) - len_foreground)
        background_clipped = background_samples[background_offset:(
            background_offset + len_foreground)]
        background_reshaped = background_clipped.reshape([1, len_foreground])

        bg_vol = np.random.uniform(0, 0.1) #may be changed
        background_mul = background_reshaped * bg_vol

        return background_mul
    
    def augment_samples(self,x, aug_factor=0):
        self.background_data = self.load_background_data()
        noisy_samples = [x]
        print(x.size())
        len_samples= x.size(3)
        for i in range(aug_factor):
            background_mul = self.get_background(len_samples)
            print(background_mul.size())
            if x.is_cuda:
                background_mul = background_mul.cuda()
            #print(x)
            background_add =  x.add(background_mul)
            #print(background_add)
            print(background_add.size())
            background_clamped = torch.clamp(background_add, -1.0, 1.0)
            noisy_samples.append(background_clamped)
        x = torch.cat(noisy_samples, dim=1)
        print(x.size())
        return x

    @torch.no_grad()
    def fit_batch_offline(self, x, class_list, tailsize=5):
        """
        Fit the NCM model to a new sample (x,y).
        :param item_ix:
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        if self.cuda:
            x = x.cuda()

        self.num_classes = x.size(0)
        n_support = x.size(1)
        self.input_shape = x.size(2)

        assert n_support>=tailsize, "Tailsize is smaller than number of examples provided"

        # inference
        xx = x.view(self.num_classes * n_support, *x.size()[2:])
        zq = self.backbone.get_embeddings(xx)
        zq = zq.view(self.num_classes, n_support, zq.size(-1))
        z_proto = zq.mean(1)

        print(zq.size(), z_proto.size())

        #########################################################
        # get more samples to compute the distributions
    #    x = self.augment_samples(x, aug_factor=2)
    #    self.num_classes = x.size(0)
    #    n_support = x.size(1)
    #    self.input_shape = x.size(2)
    #    x = x.view(self.num_classes * n_support, *x.size()[2:])
    #    zq = self.backbone.get_embeddings(x)
    #    zq = zq.view(self.num_classes, n_support, zq.size(-1))
    #    tailsize = 80
        ##########################################################

        # for each category, read meanfile, distance file, and perform weibull fitting
        weibull_model = {}
        proto_vector = []
        catId = 0
        for i,category in enumerate(class_list):
            if 'unknown' in category:
                continue
            self.word_to_index[category] = catId
        
            print('***** Category:', category, i,' *****')
            weibull_model[catId] = {}
            meantrain_vec = z_proto[i].unsqueeze(0)
            print(meantrain_vec.size(), zq[i].size())
            proto_vector.append(z_proto[i])
            
            if self.backbone.criterion.distance == 'euclidean': 
                distances = euclidean_dist(zq[i], meantrain_vec)
            elif self.backbone.criterion.distance == 'cosine': 
                #print(X.size(), self.muK.size())
                distances = 1 - F.cosine_similarity( 
                    zq[i].unsqueeze(1).expand( zq[i].size(0), *meantrain_vec.size()), 
                    meantrain_vec, dim=2)
            
            print(distances.size())
            distances = distances.squeeze().cpu().numpy()
            #meantrain_vec = meantrain_vec.squeeze().cpu().numpy()

            weibull_model[catId]['distances_%s'%self.backbone.criterion.distance] = distances
            weibull_model[catId]['mean_vec'] = meantrain_vec

            print(distances)

            mr = libmr.MR()
            tailtofit = sorted(distances)[-tailsize:]
            print(tailtofit, mr.is_valid)
            
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[catId]['weibull_model'] = mr
            print('after update:', mr.is_valid)

            catId +=1

        self.word_to_index['_unknown_'] = catId



        # update models 
        self.weibull_model = weibull_model
        self.z_proto = torch.stack(proto_vector)

    def class2torchidx(self, labels):
        label_list = []
        for item in labels:
            label_index = self.word_to_index[item]
            label_list.append([label_index])
        return torch.LongTensor(label_list)

    @torch.no_grad()
    def evaluate_batch(self, test_x, labels, return_probas=False):

        #print(self.word_to_index)
        if isinstance(labels[0], str):
            target_inds = self.class2torchidx(labels)

        if self.cuda:
            test_x = test_x.cuda()
            target_inds = target_inds.cuda()

        if self.backbone is not None:
            zq = self.backbone.get_embeddings(test_x)
        else:
            zq = test_x

        

        if self.backbone.criterion.distance == 'euclidean': 
            distances = euclidean_dist(zq, self.z_proto)
        elif self.backbone.criterion.distance == 'cosine': 
            distances =  1 - F.cosine_similarity( 
                    zq.unsqueeze(1).expand( zq.size(0), *self.z_proto.size()), 
                    self.z_proto, dim=2)


        # compute openmax
        NCLASSES = len(self.weibull_model) 
        n_batch = distances.size(0)
        #print(NCLASSES, distances)

        count = 0
        score_batch = []
        for el in range(n_batch):
            #print('oginal distance: ', distances[el], 'label:', target_inds[el].item(), labels[el])
            openmax_class_score = []
            openmax_unknown_score = 0

            feat_vec = zq[el].unsqueeze(0)
            distance_score = F.softmax (-distances[el], dim=0)
            #print('getting the score:',distances[el],distance_score)

            for categoryid in range(NCLASSES):
                # get distance between current channel and mean vector
                channel_distance = distances[el,categoryid].item()
                category_score = distance_score[categoryid].item()
                # obtain w_score for the distance and compute probability of the distance
                # being unknown wrt to mean training vector and channel distances for
                # category and channel under consideration
                wscore = self.weibull_model[categoryid]['weibull_model'].w_score(channel_distance)
                #modified_fc8_score = category_score * ( 1 - wscore)
                #openmax_class_score += [modified_fc8_score]
                #openmax_unknown_score += (category_score - modified_fc8_score)                
                

                modified_fc8_score = channel_distance * wscore
                openmax_class_score += [modified_fc8_score]
                openmax_unknown_score += (channel_distance - modified_fc8_score)    
                 
                #print(wscore, channel_distance ,modified_fc8_score, openmax_unknown_score)
            openmax_class_score += [ openmax_unknown_score ]
            score_i = torch.Tensor(openmax_class_score).unsqueeze(0)
            #score_i = F.normalize(score_i, p=1.0, dim = 1)
            score_i = F.softmax(-score_i, dim=1).cpu()

            #print('final score: ',score_i.squeeze())  
            score_batch.append(score_i)

            #exit()

        scores = torch.cat(score_batch)
        #print(scores)



        return scores, target_inds


