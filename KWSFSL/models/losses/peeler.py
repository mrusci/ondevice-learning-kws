# implementation from https://github.com/BoLiu-SVCL/meta-open/

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import euclidean_dist
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class peeler_loss(nn.Module):
    def __init__(self, args):
        super(peeler_loss, self).__init__()
        print(args)
        self.n_support = args['n_support']
        self.n_query = args['n_query']
        self.n_way_u = args['n_way_u']
        self.distance =  'euclidean'
        self.margin = args['margin']
        self.in_feats = args['z_dim']

        # define extra PEELER layers
        self.cel_all = nn.CrossEntropyLoss()
        block = BasicBlock
        self.inplanes = self.in_feats * block.expansion
        self.layer_sigs_0 = self._make_layer(block, self.in_feats, 2, stride=1)
        self.inplanes = self.in_feats * block.expansion * 2
        self.layer_sigs_1 = self._make_layer(block, self.in_feats, 2, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sm = nn.Softmax(dim=1)

        
    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def compute(self, zq, n_sample, n_class):

        # split support and query samples
        assert n_class > self.n_way_u, "Amount of unknown classes is {} must be lower than classes per episode (currently: {}) per episode".format(self.n_way_u, n_class)
        assert n_sample == (self.n_support + self.n_query), "{} samples per batch expected, got {}".format(n_sample, self.n_support + self.n_query)
        zq = zq.view(n_class, n_sample, *zq.size()[1:])

        n_class_known = n_class - self.n_way_u
        support_samples = zq[:n_class_known,:self.n_support,:].contiguous()
        query_mu = zq[:,self.n_support:self.n_support+self.n_query,:].contiguous()

        #apply PEELER
        # prepare class gauss
        mu = support_samples.mean(dim=1)
        support_samples = support_samples.view(n_class_known * self.n_support, *support_samples.size()[2:])
        query_mu = query_mu.view(n_class * self.n_query, *query_mu.size()[2:])
        batch_size, feat_size, feat_h, feat_w = query_mu.size()

        support_sigs_0 = support_samples
        support_sigs_1 = self.layer_sigs_0(support_sigs_0).mean(dim=0, keepdim=True).expand_as(support_sigs_0)
        support_sigs_1 = torch.cat((support_sigs_0, support_sigs_1), dim=1)
        support_sigs = self.layer_sigs_1(support_sigs_1)
        support_sigs=support_sigs.view(n_class_known, self.n_support, *support_sigs.size()[1:])

        # compute targets
        target_inds = torch.arange(0, n_class_known).view(n_class_known, 1, 1).expand(n_class_known, self.n_query, 1).reshape(n_class_known * self.n_query).long()
        target_inds = Variable(target_inds, requires_grad=False)
        if zq.is_cuda:
            target_inds = target_inds.cuda()

        # fewshot loss
#        print(support_samples.size(), support_sigs.size())

        support_sigs = support_sigs.mean(dim=1)
        mu_whitten = torch.mul(mu, support_sigs)
#        print(mu.size(), support_sigs.size(), query_mu.size(), mu_whitten.size())
        query_mu_whitten = torch.mul(query_mu.unsqueeze(1), support_sigs.unsqueeze(0))

        mu_whitten = self.avgpool(mu_whitten)
#        print(query_mu_whitten.size(), mu_whitten.size())

        mu_whitten = mu_whitten.view(-1, feat_size)
        query_mu_whitten = self.avgpool(query_mu_whitten.view(-1, feat_size, feat_h, feat_w))
        query_mu_whitten = query_mu_whitten.view(batch_size, -1, feat_size)
        dist_few = -torch.norm(query_mu_whitten - mu_whitten.unsqueeze(0), p=2, dim=2)
#        print(dist_few.size())
        dist_few = dist_few.view(n_class, self.n_query, n_class_known)
        dist_few_kn = dist_few[:n_class_known,:,:].contiguous()
        dist_few_un = dist_few[n_class_known:,:,:].contiguous()
        dist_few_few = dist_few_kn.view(n_class_known*self.n_query, n_class_known)
#        print(dist_few_few.size(), target_inds.size())

        l_few = self.cel_all(dist_few_few, target_inds)

        dist_few_open = dist_few_un.view(self.n_way_u*self.n_query, n_class_known)
        loss_open = F.softmax(dist_few_open, dim=1) * F.log_softmax(dist_few_open, dim=1)
        loss_open = loss_open.sum(dim=1)
        l_open = loss_open.mean()
        loss_val = l_few + l_open * 0.5
#        print(loss_val)

        return loss_val




