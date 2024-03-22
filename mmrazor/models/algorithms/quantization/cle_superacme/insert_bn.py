import torch
import torch.nn as nn

def update_running_mean_var(x, running_mean, running_var, momentum=0.9, is_first_batch=False):
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    if is_first_batch:
        running_mean = mean
        running_var = var
    else:
        running_mean = momentum * running_mean + (1.0 - momentum) * mean
        running_var = momentum * running_var + (1.0 - momentum) * var
    return running_mean, running_var

#   Record the mean and std like a BN layer but do no normalization
class BNStatistics(nn.Module):
    def __init__(self, num_features):
        super(BNStatistics, self).__init__()
        shape = (1, num_features, 1, 1)
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.zeros(shape))
        self.is_first_batch = True

    def forward(self, x):
        if self.running_mean.device != x.device:
            self.running_mean = self.running_mean.to(x.device)
            self.running_var = self.running_var.to(x.device)
        self.running_mean, self.running_var = update_running_mean_var(x, self.running_mean, self.running_var, momentum=0.9, is_first_batch=self.is_first_batch)
        self.is_first_batch = False
        return x

#   This is designed to insert BNStat layer between Conv2d(without bias) and its bias
class BiasAdd(nn.Module):
    def __init__(self, num_features):
        super(BiasAdd, self).__init__()
        self.bias = torch.nn.Parameter(torch.Tensor(num_features))
    def forward(self, x):
        return x + self.bias.view(1, -1, 1, 1)

        
        
def switch_mv2block_to_bnstat(model):
    for n, block in model.named_modules():
        if isinstance(block, nn.Sequential):
            if block.__len__()==3 and isinstance(block.__getitem__(2),nn.ReLU):
                print('switch to BN Statistics: ', n)
                conv_old=block.__getitem__(0)
                stat = nn.Sequential()
                stat.add_module('conv', nn.Conv2d(conv_old.in_channels, conv_old.out_channels,
                                                conv_old.kernel_size,
                                                conv_old.stride, conv_old.padding,
                                                conv_old.dilation,
                                                conv_old.groups, bias=False))  # Note bias=False
                stat.add_module('bnstat', BNStatistics(conv_old.out_channels))
                stat.add_module('biasadd', BiasAdd(conv_old.out_channels))  # Bias is here
                stat.conv.weight.data = conv_old.weight.data
                stat.biasadd.bias.data = conv_old.bias.data
                block.__setitem__(0,stat)
                block.__delitem__(1)
                
            elif (block.__len__()>=3) and (block.__len__()<=19) and isinstance(block.__getitem__(block.__len__()-1),nn.Identity):
                print('switch to BN Statistics: ', n)
                conv_old=block.__getitem__(block.__len__()-2)
                stat = nn.Sequential()
                stat.add_module('conv', nn.Conv2d(conv_old.in_channels, conv_old.out_channels,
                                                conv_old.kernel_size,
                                                conv_old.stride, conv_old.padding,
                                                conv_old.dilation,
                                                conv_old.groups, bias=False))  # Note bias=False
                stat.add_module('bnstat', BNStatistics(conv_old.out_channels))
                stat.add_module('biasadd', BiasAdd(conv_old.out_channels))  # Bias is here
                stat.conv.weight.data = conv_old.weight.data
                stat.biasadd.bias.data = conv_old.bias.data
                b_n=block.__len__()
                block.__setitem__(b_n-2,stat)
                block.__delitem__(b_n-1)

def switch_bnstat_to_convbn(model):
    for n, block in model.named_modules():
        if isinstance(block, nn.Sequential):
            if block.__len__()==3 and isinstance(block.__getitem__(2),BiasAdd):
                print('switch to ConvBN: ', n)
                conv_=block.__getitem__(0)
                conv = nn.Conv2d(conv_.in_channels, conv_.out_channels,
                                conv_.kernel_size,
                                conv_.stride, conv_.padding,
                                conv_.dilation,
                                conv_.groups, bias=False)
                bn = nn.BatchNorm2d(conv_.out_channels)
                bn.running_mean = block.bnstat.running_mean.squeeze()  # Initialize the mean and var of BN with the statistics
                bn.running_var = block.bnstat.running_var.squeeze()
                std = (bn.running_var + bn.eps).sqrt()
                conv.weight.data = conv_.weight.data
                bn.weight.data = std
                bn.bias.data = block.biasadd.bias.data + bn.running_mean  # Initialize gamma = std and beta = bias + mean
                
                block.__setitem__(0,conv)
                block.__setitem__(1,bn)
                block.__delitem__(2)
                
                
def switch_mv2block_to_bnstat_mm(model,instance_type):
    for n, block in model.named_modules():
        if isinstance(block, instance_type):
            print('switch to BN Statistics: ', n)
            conv_old=block.conv
            stat = nn.Sequential()
            stat.add_module('conv', nn.Conv2d(conv_old.in_channels, conv_old.out_channels,
                                            conv_old.kernel_size,
                                            conv_old.stride, conv_old.padding,
                                            conv_old.dilation,
                                            conv_old.groups, bias=False))  # Note bias=False
            stat.add_module('bnstat', BNStatistics(conv_old.out_channels))
            stat.add_module('biasadd', BiasAdd(conv_old.out_channels))  # Bias is here
            stat.conv.weight.data = conv_old.weight.data
            stat.biasadd.bias.data = conv_old.bias.data
            block.conv=stat
            
def switch_bnstat_to_convbn_mm(model,instance_type):
    for n, block in model.named_modules():
        if isinstance(block, instance_type):
            print('switch to ConvBN: ', n)
            fakeconv=block.conv
            conv_=fakeconv.__getitem__(0)
            conv = nn.Conv2d(conv_.in_channels, conv_.out_channels,
                            conv_.kernel_size,
                            conv_.stride, conv_.padding,
                            conv_.dilation,
                            conv_.groups, bias=False)
            bn = nn.BatchNorm2d(conv_.out_channels)
            bn.running_mean = fakeconv.__getitem__(1).running_mean.squeeze()  # Initialize the mean and var of BN with the statistics
            bn.running_var = fakeconv.__getitem__(1).running_var.squeeze()
            std = (bn.running_var + bn.eps).sqrt()
            conv.weight.data = conv_.weight.data
            bn.weight.data = std
            bn.bias.data = fakeconv.__getitem__(2).bias.data + bn.running_mean  # Initialize gamma = std and beta = bias + mean
            
            block.conv=conv
            block.bn=bn