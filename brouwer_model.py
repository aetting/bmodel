import torch, random
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from util_brouwer import *
import copy

# randomize weights in [-.25,.25]
# learning rate of 0.2, which was scaled down to 0.11 with a factor of 0.95 after each 700 epochs (that is, after each 10% interval of the total epochs; 0:2 0:9510 % 0:11)
# momentum coefficient was set to a constant of 0.9. Finally, we used a zero error radius of 0.1, such that no error was backpropagated when the dif- ference between the produced activity level yj of a unit j and the desired activity level dj of this unit was smaller than 0.1,


class testNet(nn.Module):

    def __init__(self,vocab_size,emb_size,labelnum,context_size,intersize,input='loc',output='cat'):
        super(testNet, self).__init__()
#         self.logist = nn.Sigmoid()
#         self.soft = nn.Softmax()
#         self.retr = nn.Linear(vocab_size,intersize,bias=False)
#         self.retr1 = nn.RNNCell(vocab_size,context_size,bias=False)
        if input == 'loc':
            self.retr1 = nn.Linear(vocab_size+context_size,context_size,bias=False)
        elif input == 'dist':
            self.retr1 = nn.Linear(emb_size+context_size,context_size,bias=False)
        else: raise Exception('Invalid input type!')
        
        if output == 'cat':
            self.outnonlin = nn.Softmax()
            self.retr2 = nn.Linear(context_size,labelnum,bias=False)
        elif output == 'dist':
            self.outnonlin = nn.Sigmoid()
            self.retr2 = nn.Linear(context_size,3*emb_size,bias=False)
        else: raise Exception('Invalid output type!')
        
    def forward(self,x,h):
#         x = self.logist(self.retr(x))
#         h = self.retr1(x,h)
        x = torch.cat((x,h),dim=1)
        h = F.tanh(self.retr1(x).add(1.0))
        x = self.outnonlin(self.retr2(h).add(1.0))
        return x,h,None
        
class NetInteg(nn.Module):

    def __init__(self,vocab_size,emb_size,context_size,labelnum,input='dist',output = 'dist'):
        super(NetInteg, self).__init__() #TODO understand this better
#         self.nonlin1 = nn.Tanh()
#         self.nonlin2 = nn.Sigmoid()
        if input == 'dist':
            self.integ = nn.Linear(context_size+emb_size,context_size, bias=False)
        elif input == 'loc':
            self.integ = nn.Linear(context_size+vocab_size,context_size, bias=False)
        else: raise Exception('Invalid input type!')
        
        if output == 'loc':
            self.integ_out = nn.Linear(context_size,3*vocab_size, bias=False)
        elif output == 'dist': 
            self.integ_out = nn.Linear(context_size,3*emb_size, bias=False)
        elif output == 'lm':
            self.integ_out = nn.Linear(context_size,vocab_size, bias=False)
        elif output == 'cat':
            self.integ_out = nn.Linear(context_size,labelnum, bias=False)
        else: raise Exception('Invalid output type!')
        
    def forward(self,x,integ_context):
        #concatenate word representation with integ_context
        x = torch.cat((x,integ_context),dim=1)
        #integration layer: linear with bias of 1, sent through sigmoid activation
        integ_context = F.sigmoid(self.integ(x).add(1))
        #save integration layer output for input to next time step
#         integ_context = x.clone()
        #integration output layer: linear with bias of 1, sent through sigmoid activation
        x = F.sigmoid(self.integ_out(integ_context).add(1))
        
        return x,integ_context,None

class NetFull(nn.Module):

    def __init__(self,vocab_size,emb_size,context_size,retrieval_size,labelnum,output = 'dist'):
        super(NetFull, self).__init__() #TODO understand this better
#         self.logist = nn.Sigmoid()
#         self.soft = nn.Softmax()
        self.retr = nn.Linear(context_size+vocab_size,retrieval_size, bias=False)
        self.retr_out = nn.Linear(retrieval_size,emb_size, bias=False)
        
        self.integ = nn.Linear(context_size+emb_size,context_size, bias=False)
        
        if output == 'loc':
            self.integ_out = nn.Linear(context_size,3*vocab_size, bias=False)
        elif output == 'dist': 
            self.integ_out = nn.Linear(context_size,3*emb_size, bias=False)
        elif output == 'lm':
            self.integ_out = nn.Linear(context_size,vocab_size, bias=False)
        elif output == 'cat':
            self.integ_out = nn.Linear(context_size,labelnum, bias=False)
        else: raise Exception('Invalid output type!')
            
    def forward(self,x,integ_context_prev):
        #concatenate integ_context with input word

        x = torch.cat((x,integ_context_prev),dim=1)
        #retrieval layer: linear with bias of 1, sent through sigmoid activation

        retr_layer = F.sigmoid(self.retr(x).add(1))
        #retrieval output layer: linear with bias of 1, sent through sigmoid activation

#         retr_layer = x.clone()

        x = F.sigmoid(self.retr_out(retr_layer).add(1))
        #concatenate retrieval output with integ_context

        x = torch.cat((x,integ_context_prev),dim=1)
        #integration layer: linear with bias of 1, sent through sigmoid activation

        integ_context = F.sigmoid(self.integ(x).add(1))
        #save integration layer output for input to next time step

#         integ_context = x.clone()

        #integration output layer: linear with bias of 1, sent through sigmoid activation
        x = F.sigmoid(self.integ_out(integ_context).add(1))
        
        return x,integ_context,retr_layer