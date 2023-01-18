import sys
import os
# import numpy as npf
# import autograd.numpy as np
import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *


# TODO: construct a net from the architecture in the config yaml

class identity_net(torch.nn.Module):
    def __init__(self, num_orders):
        super(identity_net, self).__init__()
        
        self.identity = torch.nn.Identity()
        
        # This is just here so any NN optimizers won't freak out and say there's no model.parameters()
        self.conv = torch.nn.Conv2d( num_orders, 1, 3, padding=1 )
    
    def forward(self, x):
        
        #! These operations might not be allowed / might not be tracked in forward()!
        # Naively sum intensities from all orders here.
        # x = torch.sum(x, 1) 
        # # Naively normalize to [0,1]  #! Especially this one!!!
        # original_shape = x.size()
        # x = x.view(x.size(0), -1)
        # x -= x.min(1, keepdim=True)[0]
        # x /= x.max(1, keepdim=True)[0]
        # x = x.view(original_shape)
        
        return x
        

class simple_net(torch.nn.Module):
    def __init__(self, num_orders):
        super(simple_net, self).__init__()

        self.conv1 = torch.nn.Conv2d( num_orders, num_orders, 3, padding=1 )
        self.conv2 = torch.nn.Conv2d( num_orders, num_orders, 3, padding=1 )
        self.conv = torch.nn.Conv2d( num_orders, 1, 3, padding=1 )

    def forward(self, x):
        # x = self.conv1( ( x - 0.0168 ) / 0.0887 )
        # x = self.conv2( x )
        # return self.conv( x )

        return self.conv( ( x - 0.0168 ) / 0.0887 )

        # return ( ( x[ :, 0 ] - 0.0168 ) / 0.0887 )


class Dense_Net(nn.Module):
    def __init__(self, hidden_layer_neurons):
        '''Inherits the Module class from torch.nn and overwrites hyperparameters with user-defined ones.'''
        super(Dense_Net, self).__init__()
        # NN layers from torch.nn contain trainable parameters
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)        # 2D convolutional layers
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()                    # Dropout layer
        # self.fc1 = nn.Linear(320, 50)                       # Fully connected (linear) layers
        # self.fc2 = nn.Linear(50, 10)
        
        if hidden_layer_neurons == 0:
            self.NoIntermediateLayer = True
        else:   self.NoIntermediateLayer = False
        
        self.fc0 = nn.Linear(28*28, 10)
        self.fc1 = nn.Linear(28*28, hidden_layer_neurons)
        # self.fc2 = nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        # self.fc3 = nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        self.fc4 = nn.Linear(hidden_layer_neurons, 10)
        
class Dense_Net_OpticalFront(nn.Module):
    def __init__(self, hidden_layer_neurons):
        '''Inherits the Module class from torch.nn and overwrites hyperparameters with user-defined ones.'''
        super(Dense_Net_OpticalFront, self).__init__()
        # NN layers from torch.nn contain trainable parameters
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)        # 2D convolutional layers
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()                    # Dropout layer
        # self.fc1 = nn.Linear(320, 50)                       # Fully connected (linear) layers
        # self.fc2 = nn.Linear(50, 10)
        
        self.tmask = nn.Parameter(torch.ones(1,28,28, requires_grad=True, dtype=torch.complex64))
        
        if hidden_layer_neurons == 0:
            self.NoIntermediateLayer = True
        else:   self.NoIntermediateLayer = False
        
        self.fc0 = nn.Linear(28*28, 10)
        self.fc1 = nn.Linear(28*28, hidden_layer_neurons)
        # self.fc2 = nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        # self.fc3 = nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        self.fc4 = nn.Linear(hidden_layer_neurons, 10)
        
    def forward(self, x):
        '''Defines the way we compute our output using the given layers and functions.'''
        # Functional layers from torch.nn.functional are purely functional
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        
        x = utility.torch_2dft(x)
        # print(x.size())
        x = self.tmask*x
        # print(x.size())
        x = torch.real(utility.torch_2dift(x))        #todo: Implement support for complex tensors i.e. we shouldn't be taking the IFT
        
        
        x = x.view(-1, 28*28)
        # x = torch.view_as_real(x)       # Necessary for now because Linear layers don't support ComplexFloats: 12/06/22
        if self.NoIntermediateLayer:
            x = self.fc0(x)
        else:
            x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            # x = F.relu(self.fc3(x))
            x = self.fc4(x)
            
        return F.log_softmax(x)

class Conv_Net(nn.Module):
    def __init__(self, num_conv_channels, num_kernels, num_fc_neurons):
        '''Inherits the Module class from torch.nn and overwrites hyperparameters with user-defined ones.'''
        super(Conv_Net, self).__init__()
        # NN layers from torch.nn contain trainable parameters
        self.conv1 = nn.Conv2d(1, num_conv_channels, kernel_size=num_kernels, stride=num_kernels)         # 2D convolutional layers
        self.conv1_drop = nn.Dropout2d()                    # Dropout layer
    
        self.conv_output = int((28 - 1*(num_kernels-1) - 1)/num_kernels + 1)
        self.pooled_output = int((self.conv_output - 1*(2-1) - 1)/2 + 1)
        self.flattened_width = int(num_conv_channels*self.pooled_output**2)
        
        self.fc1 = nn.Linear(self.flattened_width, num_fc_neurons)                       # Fully connected (linear) layers
        self.fc2 = nn.Linear(num_fc_neurons, 10)
        
        
    def forward(self, x):
        '''Defines the way we compute our output using the given layers and functions.'''
        # Functional layers from torch.nn.functional are purely functional
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))                        # max_pool2d has stride = kernel by default
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        
        #! Calculate FLOPs from here:   https://www.thinkautonomous.ai/blog/deep-learning-optimization/#flops_calc
        # Valentine paper is 2 x 9 x (3x3) x (8x8) i.e. n_kernel = 9, kernel_shape = (3x3), output_shape = (8x8)
        # print('Initial size:')
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        x = self.conv1_drop(x)
        # print(x.size())
        x = F.max_pool2d(x,2)
        # print(x.size())
        x = F.relu(x)
        # print(x.size())
        x = x.reshape(-1, self.flattened_width)
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = F.dropout(x, training=self.training)
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        
        # x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)),2))
        # x = x.view(-1,144)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
            
        return F.log_softmax(x)
    

class Conv_Net_OpticalFront(nn.Module):
    def __init__(self, num_conv_channels, num_kernels, num_fc_neurons):
        '''Inherits the Module class from torch.nn and overwrites hyperparameters with user-defined ones.'''
        super(Conv_Net_OpticalFront, self).__init__()
        # NN layers from torch.nn contain trainable parameters
        
        self.tmask = nn.Parameter(torch.ones(1,28,28, requires_grad=True, dtype=torch.complex64))
        
        self.conv1 = nn.Conv2d(1, num_conv_channels, kernel_size=num_kernels, stride=num_kernels)         # 2D convolutional layers
        self.conv1_drop = nn.Dropout2d()                    # Dropout layer
    
        self.conv_output = int((28 - 1*(num_kernels-1) - 1)/num_kernels + 1)
        self.pooled_output = int((self.conv_output - 1*(2-1) - 1)/2 + 1)
        self.flattened_width = int(num_conv_channels*self.pooled_output**2)
        
        self.fc1 = nn.Linear(self.flattened_width, num_fc_neurons)                       # Fully connected (linear) layers
        self.fc2 = nn.Linear(num_fc_neurons, 10)
        
        
    def forward(self, x):
        '''Defines the way we compute our output using the given layers and functions.'''
        # Functional layers from torch.nn.functional are purely functional
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))                        # max_pool2d has stride = kernel by default
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        
        
        x = utility.torch_2dft(x)
        # print(x.size())
        x = self.tmask*x
        # print(x.size())
        x = torch.real(utility.torch_2dift(x))        #todo: Implement support for complex tensors i.e. we shouldn't be taking the IFT
        
        
        #! Calculate FLOPs from here:   https://www.thinkautonomous.ai/blog/deep-learning-optimization/#flops_calc
        # Valentine paper is 2 x 9 x (3x3) x (8x8) i.e. n_kernel = 9, kernel_shape = (3x3), output_shape = (8x8)
        # print('Initial size:')
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        x = self.conv1_drop(x)
        # print(x.size())
        x = F.max_pool2d(x,2)
        # print(x.size())
        x = F.relu(x)
        # print(x.size())
        x = x.reshape(-1, self.flattened_width)
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = F.dropout(x, training=self.training)
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        
        # x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)),2))
        # x = x.view(-1,144)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
            
        return F.log_softmax(x)