# adapted from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/angleproto.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import euclidean_dist
from torch.autograd import Variable

class angular_proto_loss(nn.Module):
    def __init__(self, args, init_w=10.0, init_b=-5.0):
        super(angular_proto_loss, self).__init__()

        self.n_support = args['n_support']
        self.n_query = args['n_query']
        self.distance =  'cosine'
        self.margin = args['margin']
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def compute(self, zq, n_sample, n_class):

        # split support and query samples
        assert n_sample == (self.n_support + self.n_query), "{} samples per batch expected, got {}".format(n_sample, self.n_support + self.n_query)
        zq = zq.view(n_class, n_sample, *zq.size()[1:])
        support_samples = zq[:,:self.n_support,:].contiguous()
        query_samples = zq[:,self.n_support:self.n_support+self.n_query,:].contiguous()
        
        # compute protoptypes using support vectors
        support_proto = support_samples.mean(1)
        query_samples = query_samples.view(n_class*self.n_query, *query_samples.size()[2:])


        input1 = query_samples.unsqueeze(1).expand( n_class*self.n_query, n_class, query_samples.size(-1))
        score = F.cosine_similarity(input1, support_proto, dim=2)
        if self.margin > 0:
            for cl in range(n_class):
                score[self.n_query*cl:self.n_query*cl+self.n_query,cl].add_(-self.margin)

        # apply learned scaling
        torch.clamp(self.w, 1e-6)
        score = score * self.w + self.b

        # compute targets
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, self.n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        if zq.is_cuda:
            target_inds = target_inds.cuda()
        
        # compute crossentropy loss
        log_p_y = F.log_softmax(score, dim=1)
        log_p_y = log_p_y.view(n_class, self.n_query, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        return loss_val

