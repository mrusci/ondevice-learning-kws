import torch.nn as nn
from models.encoder.baseUtil import Flatten
import math

class DSCNN(nn.Module):
    
    def __init__(self, t_dim, f_dim, model_size_info, padding_0, last_norm=True, return_feat_maps=False ):
        super(DSCNN, self).__init__()
        self.input_features = [t_dim,f_dim]
        self.return_feat_maps = return_feat_maps

        num_layers = model_size_info[0]
        conv_feat = [0 for x in range(num_layers)]
        conv_kt = [0 for x in range(num_layers)]
        conv_kf = [0 for x in range(num_layers)]
        conv_st = [0 for x in range(num_layers)]
        conv_sf = [0 for x in range(num_layers)]
        i=1
        for layer_no in range(0,num_layers):
            conv_feat[layer_no] = model_size_info[i]
            i += 1
            conv_kt[layer_no] = model_size_info[i]
            i += 1
            conv_kf[layer_no] = model_size_info[i]
            i += 1
            conv_st[layer_no] = model_size_info[i]
            i += 1
            conv_sf[layer_no] = model_size_info[i]
            i += 1
            
            
        ds_cnn_layers = []
        
        for layer_no in range(0,num_layers):
            num_filters = conv_feat[layer_no]
            kernel_size = (conv_kt[layer_no],conv_kf[layer_no])
            stride = (conv_st[layer_no],conv_sf[layer_no])


            t_dim_b = t_dim
            f_dim_b = f_dim

            t_dim = math.ceil(t_dim/float(conv_st[layer_no]))
            f_dim = math.ceil(f_dim/float(conv_sf[layer_no]))

            t_pad  = (t_dim - 1) * conv_st[layer_no] + conv_kt[layer_no] - t_dim_b
            t_pad_l = math.ceil(t_pad/2)
            t_pad_r = t_pad - t_pad_l
            f_pad  = (f_dim - 1) * conv_sf[layer_no] + conv_kf[layer_no] - f_dim_b
            f_pad_l = math.ceil(f_pad/2)
            f_pad_r = f_pad - f_pad_l

            padding = (f_pad_l, f_pad_r, t_pad_l, t_pad_r  )

            if layer_no==0:
                # P = ((S-1)*W-S+F)/2, with F = filter size, S = stride, W = input size
                ds_cnn_layers.append( nn.ZeroPad2d( padding ) )
                ds_cnn_layers.append( nn.Conv2d(in_channels = 1, out_channels = num_filters, kernel_size = kernel_size, stride = stride, bias = True) )
                ds_cnn_layers.append( nn.BatchNorm2d(num_filters) )
                ds_cnn_layers.append( nn.ReLU() )
            else:
                ds_cnn_layers.append( nn.ZeroPad2d(padding) )
                ds_cnn_layers.append( nn.Conv2d(in_channels = num_filters, out_channels = num_filters, kernel_size = kernel_size, stride = stride, groups = num_filters, bias = True) )
                ds_cnn_layers.append( nn.BatchNorm2d(num_filters) )
                ds_cnn_layers.append( nn.ReLU() )
                ds_cnn_layers.append( nn.Conv2d(in_channels = num_filters, out_channels = num_filters, kernel_size = (1, 1), stride = (1, 1), bias = True) )
                if (last_norm== True) or (layer_no < num_layers-1):
                    ds_cnn_layers.append( nn.BatchNorm2d(num_filters) )
                    ds_cnn_layers.append( nn.ReLU() )
                elif (last_norm== 'Layer'):
                    ds_cnn_layers.append( nn.LayerNorm([num_filters, t_dim_b, f_dim_b], elementwise_affine=False) )
                elif (last_norm== 'Batch'):
                    ds_cnn_layers.append( nn.BatchNorm2d(num_filters) )

        self.dscnn = nn.Sequential(*ds_cnn_layers)
        self.embedding_features = num_filters

        self.avgpool = nn.AvgPool2d(kernel_size=(t_dim, f_dim), stride=1) 
        self.flatten = Flatten() 
        
    def forward(self, x):
        x = self.dscnn(x)
        if self.return_feat_maps:
            return x
        x = self.avgpool(x)
        x = self.flatten(x)
        return x

        
            
# DSCNN_SMALL
model_size_info_DSCNNS = [5, 64, 10, 4, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1]
padding_0_DSCNNS = (6,1)

def DSCNNS(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNS, padding_0_DSCNNS )

def DSCNNS_NONORM(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNS, padding_0_DSCNNS, last_norm=False  )

def DSCNNS_LAYERNORM(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNS, padding_0_DSCNNS, last_norm='Layer'  )

def DSCNNS_BATCHNORM(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNS, padding_0_DSCNNS  , last_norm='Batch'  )


# DSCNN_MEDIUM
model_size_info_DSCNNM = [5, 172, 10, 4, 2, 1, 172, 3, 3, 2, 2, 172, 3, 3, 1, 1, 172, 3, 3, 1, 1, 172, 3, 3, 1, 1]
padding_0_DSCNNM = (5,1)

def DSCNNM(x_dim):
    return DSCNN(x_dim[1], x_dim[2],model_size_info_DSCNNM, padding_0_DSCNNM  )

def DSCNNM_LAYERNORM(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNM, padding_0_DSCNNM  , last_norm='Layer'  )

def DSCNNM_BATCHNORM(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNM, padding_0_DSCNNM  , last_norm='Batch'  )


# DSCNN_LARGE
model_size_info_DSCNNL = [6, 276, 10, 4, 2, 1, 276, 3, 3, 2, 2, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1]
padding_0_DSCNNL = (5,2)
padding_0_DSCNNL = (5,4,2,1)

def DSCNNL(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNL, padding_0_DSCNNL  )

def DSCNNL_NONORM(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNL, padding_0_DSCNNL , last_norm=False  )

def DSCNNL_LAYERNORM(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNL, padding_0_DSCNNL  , last_norm='Layer'  )

def DSCNNL_BATCHNORM(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNL, padding_0_DSCNNL  , last_norm='Batch'  )


# for peeler
def DSCNNS_PEELER(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNS, padding_0_DSCNNS, return_feat_maps=True )
def DSCNNL_PEELER(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNL, padding_0_DSCNNL, return_feat_maps=True )
