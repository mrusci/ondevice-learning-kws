# adapted from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/amsoftmax.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class am_softmax(nn.Module):
    def __init__(self, args, scale=64):
        super(am_softmax, self).__init__()
        self.distance =  'cosine'
        self.margin = args['margin']
        self.scale = scale
        self.in_feats = args['z_dim']
        self.n_class = args['n_classes']
        self.W = torch.nn.Parameter(torch.randn(self.in_feats, self.n_class), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AMSoftmax m=%.3f s=%.3f'%(self.margin,self.scale))

    def compute(self, x, label):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
#        print('Input:', x.size(), label.size())
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        if self.margin > 0:
            delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.margin)
            if x.is_cuda: delt_costh = delt_costh.cuda()
#            print('Internals:', costh.size(), delt_costh.size(), delt_costh)
            costh_m = costh - delt_costh
        else:
            costh_m = costh
        costh_m_s = self.scale * costh_m
        loss    = self.ce(costh_m_s, label)
#        print('loss:',loss.size(), loss)
        _, predicted = torch.max(costh_m_s.detach(), 1)
        correct = torch.eq(predicted, label).float()
        acc_val = correct.mean()
#        print(acc_val)

        return loss, {
            'loss': loss.item(),
            'acc': acc_val.item(),
            'score_mean': 0,
            'score_corr': 0,
            'score_wrong': 0
        }


