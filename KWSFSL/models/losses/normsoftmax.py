import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import euclidean_dist
from torch.autograd import Variable

class prototypical_loss(nn.Module):
    def __init__(self, args):
        super(prototypical_loss, self).__init__()
        self.n_support = args['n_support']
        self.n_query = args['n_query']
        self.distance =  'euclidean'
        self.margin = args['margin']

    def compute(self, zq, n_sample, n_class):

        # split support and query samples
        assert n_sample == (self.n_support + self.n_query), "{} samples per batch expected, got {}".format(n_sample, self.n_support + self.n_query)
        zq = zq.view(n_class, n_sample, *zq.size()[1:])
        support_samples = zq[:,:self.n_support,:].contiguous()
        query_samples = zq[:,self.n_support:self.n_support+self.n_query,:].contiguous()
        
        # compute protoptypes using support vectors
        support_proto = support_samples.mean(1)
        query_samples = query_samples.view(n_class*self.n_query, *query_samples.size()[2:])

        dists = euclidean_dist(query_samples, support_proto)
        score = - dists
        log_p_y = F.log_softmax(score, dim=1)
        log_p_y = log_p_y.view(n_class, self.n_query, -1)

        # compute targets
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, self.n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        if zq.is_cuda:
            target_inds = target_inds.cuda()
        
        # compute loss
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        return loss_val

