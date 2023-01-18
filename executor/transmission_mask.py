## End-to-end Optimization of Image Classifier
# Input argument (1): YAML filename
# Input argument (2): run mode [String]: 'eval' or 'opt'
# [OPTIONAL] Input argument (3): new data folder address to override the one in YAML file

#* Import all modules

# Photonics
import grcwa
grcwa.set_backend('autograd')

# Math and Autograd
import numpy as npf
import autograd.numpy as np
from autograd import grad

try:
	import nlopt
	NL_AVAILABLE = True
except ImportError:
	NL_AVAILABLE = False
if NL_AVAILABLE == False:
	raise Exception('Please install NLOPT')

import scipy
from scipy.ndimage import gaussian_filter
from scipy import optimize as scipy_optimize

# Neural Network
import torch
import torch.fft
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# System, Plotting, etc.
import os
import pickle
import sys
import time
import copy
from types import SimpleNamespace
import matplotlib.pyplot as plt

# Parallel Computing
import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

comm = MPI.COMM_WORLD
resource_size = MPI.COMM_WORLD.Get_size()
processor_rank = MPI.COMM_WORLD.Get_rank()


# Custom Classes and Imports
import DeviceGeometry
import utility
import networks

print("All modules loaded.")
sys.stdout.flush()


##############################################################################################

#* Handle Input Arguments and Parameters

if len( sys.argv ) < 3:
	print( "Usage: python " + sys.argv[ 0 ] + " [ parameters filename ] [ \'eval\' or \'opt\' ] { override data folder }" )
	sys.exit( 1 )

# Load parameters[] from pickle file created by yaml_to_parameters.py
parameters_filename = sys.argv[ 1 ]
parameters = None
with open( parameters_filename, 'rb' ) as parameters_file_handle:
	parameters = pickle.load( parameters_file_handle )
 
 # Duplicate stdout to file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(parameters[ "data_folder" ],"logfile.txt"), "a")
        
        self.log.write( "---Log for Lumerical Processing Sweep---\n" )
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    
sys.stdout = Logger()

# Determine run mode from bash script input arg
mode = sys.argv[ 2 ]
optimize = True
if not ( ( mode == 'eval' ) or ( mode == 'opt' ) ):
	print( 'Unrecognized mode!' )
	sys.exit( 1 )

# Override data folder given in YAML? If yes, third argument should be new data folder address
should_override_data_folder = False
override_data_folder = None
if len( sys.argv ) == 4:
	should_override_data_folder = True
	override_data_folder = sys.argv[ 3 ]

print("Input arguments processed.")
sys.stdout.flush()


#* Read in all the parameters!

p = SimpleNamespace(**parameters)       #! Don't use 'p' for ANYTHING ELSE!
projects_directory_location = parameters['data_folder']

#* Configure modules based on parameters
torch.manual_seed(parameters['random_seed'])
torch.backends.cudnn.enabled = parameters.get('torch_backend_cudnn_enabled')        # Disables nondeterministic algorithms for cuDNN

##############################################################################################

#* DataLoaders for the dataset

# 0.1307 and 0.3081 are the mean and stdev for the MNIST dataset respectively.
train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=p.batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=p.batch_size_test, shuffle=True)


#* Define train and test functions for the networks contained in networks.py

def train(epoch):
    network.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # Batch idx identifies which batch in the training set; 
        # data is the entire batch of training data; target is what everything in the batch is supposed to be
        
        optimizer.zero_grad()                   # Manually set all gradients to zero; PyTorch accumulates gradients by default
        
        #! Image FFT goes here
        
        output = network(data)                  # Forward pass
        
        # Get the negative log-likelihood loss between output and ground truth.
        loss = F.nll_loss(output, target)
        # Collect the gradients...
        loss.backward()
        # For backpropagation.
        optimizer.step()
        
        if batch_idx % p.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                            epoch, batch_idx * len(data), len(train_loader.dataset),
                                            100. * batch_idx / len(train_loader), loss.item()
                                            )
                  )
            train_losses.append(loss.item())
            train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            
            # Create lists for saving training and testing losses
            # NN modules and optimizers can save and load internal state using .state_dict()
            
            try:
                torch.save(network.state_dict(), MODEL_PATH)
                torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
            except Exception as ex:
                # __builtins__, my_shelf, and imported modules can not be shelved.
                # print('ERROR shelving: {0}'.format(key))
                # template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                # message = template.format(type(ex).__name__, ex.args)
                # print(message)
                pass

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():        # Avoid storing the computations done producing the output of our network in the computation graph
        
        # example = enumerate(test_loader)
        # batch_idx, (example_data, example_targets) = next(example)
        # test_image = example_data[28:29, 0:1]
        test_image = copy.deepcopy(test_image_template)
        
        img_before = np.squeeze(test_image.detach().numpy())
        test_image = utility.torch_2dft(test_image)
        test_image = network.tmask*test_image
        test_image = torch.real(utility.torch_2dift(test_image))        #todo: Implement support for completest_image tensors i.e. we shouldn't be taking the IFT
        img_after_tmask = np.squeeze(test_image.detach().numpy())
        #img_after_conv = img_after_tmask
        # test_image = network.conv1(test_image)
        # test_image = network.conv1_drop(test_image)
        # img_after_conv = np.squeeze(test_image.detach().numpy())
        
        fig = plt.figure
        plt.set_cmap("gray")
        ax1 = plt.subplot(121)
        p1 = plt.imshow(img_before, cmap='gray')
        plt.axis("off")
        ax2 = plt.subplot(122)
        p2 = plt.imshow(img_after_tmask, cmap='gray')
        plt.axis("off")
        # ax3 = plt.subplot(133)
        # p3 = plt.imshow(img_after_conv, cmap='gray')
        plt.axis("off")
        plt.colorbar(p1,ax=ax1,fraction=0.046, pad=0.04)
        plt.colorbar(p2,ax=ax2,fraction=0.046, pad=0.04)
        # plt.colorbar(p3,ax=ax3,fraction=0.046, pad=0.04)
        plt.savefig(os.path.join(projects_directory_location, f'imagesmask_epoch{epoch_count}'),
            bbox_inches='tight')
        plt.close()
        
        
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    accuracy.append(float(100. * correct / len(test_loader.dataset)))
    
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                                        test_loss, correct, len(test_loader.dataset),
                                        100. * correct / len(test_loader.dataset))
          )
    
    
    


# Dense NN
# hidden_layer_neurons = np.append(np.array([1,2]), 
#                                  np.ceil(np.power(1.2,np.arange(6,24,1.5))).astype(int)) # np.power(2, np.arange(1,8))
# hidden_layer_neurons = np.array([2,3,4])
hidden_layer_neurons = np.array([35])
# hidden_layer_neurons = np.array([3])

# Conv. NN
# num_channels = np.array([1,2,3,4,5,6,7,8]) #np.power(np.arange(3,9,1),2)
# num_kernels = np.array([1,2,3])
num_channels = np.array([9]) 
num_kernels = np.array([3])
flops = []
accuracies = []

# For-loop for varying FLOPs
for neuron_idx, neuron_size in enumerate(hidden_layer_neurons):
    for kernel_idx, kernel_size in enumerate(num_kernels):
        for channel_idx, channel_size in enumerate(num_channels):

            #* Perform training loop
            # Initialize network and optimizer.
            # network = networks.Dense_Net_OpticalFront(neuron_size)
            network = networks.Conv_Net_OpticalFront(channel_size, kernel_size, neuron_size)
            # Optimizer implements stochastic gradient descent using parameters defined at start of code.
            # optimizer = optim.SGD(network.parameters(), lr=learning_rate,
            #                     momentum=momentum)
            optimizer = optim.Adam(network.parameters(), lr=p.learning_rate)

            train_losses = []
            train_counter = []
            test_losses = []
            test_counter = [i*len(train_loader.dataset) for i in range(p.num_training_epochs + 1)]
            accuracy = []
            epoch_count = 0
            
            # num_flops = 2*(kernel_size**2 * channel_size * network.conv_output**2)    # Convolutional layer FLOPs
            # num_flops += 2*(144*neuron_size + neuron_size*10)                         # Dense layer FLOPs (for conv)
            num_flops = 2*(28**2*neuron_size + neuron_size*10)                        # Fully Dense Layer FLOPs
            print(f'Conv Network ' + \
                    # f'{kernel_idx*len(num_kernels)+channel_idx}/{len(num_channels)*len(num_kernels)}' + \
                    f'{neuron_idx} / {len(hidden_layer_neurons)}' + \
                       f' -- {channel_size} Channels, {kernel_size}x{kernel_size} Kernel, {neuron_size} Neurons -- \
                            FLOPs approx. {num_flops}')


            example = enumerate(test_loader)
            batch_idx, (example_data, example_targets) = next(example)
            # test_image_template = example_data[28:29, 0:1]
            test_image_template = example_data[30:31, 0:1]
            
            test()          # Initial accuracy/loss with randomly initialized network params.
            
            for epoch in range(1, p.num_training_epochs + 1):
                epoch_count += 1
                train(epoch)
                test()
                
                fig = plt.figure
                plt.set_cmap("gray")
                ax1 = plt.subplot(121)
                p1 = plt.imshow(abs(network.state_dict()['tmask'][0]), 
                                cmap='gray', vmin=0.5, vmax=1.8)
                plt.axis("off")
                ax2 = plt.subplot(122)
                p2 = plt.imshow(np.arctan2(np.imag(network.state_dict()['tmask'][0]), 
                                        np.real(network.state_dict()['tmask'][0])
                                        ),
                                cmap='gray')
                plt.axis("off")
                plt.colorbar(p1,ax=ax1,fraction=0.046, pad=0.04)
                plt.colorbar(p2,ax=ax2,fraction=0.046, pad=0.04)
                plt.savefig(os.path.join(projects_directory_location,f'epochs\\tmask_opt_epoch{epoch}.png'),
                    bbox_inches='tight')
                plt.close()
                
            flops.append(num_flops)
            accuracies.append(accuracy)

np_accuracies = np.array(accuracies)
np.save(os.path.join(projects_directory_location, 'num_dense_neurons.npy'), num_channels)
np.save(os.path.join(projects_directory_location, 'flops.npy'), flops)
np.save(os.path.join(projects_directory_location, 'accuracies.npy'), np_accuracies)


#* Plot training curve

marker_style = dict(linestyle='-', linewidth=2.2, marker='o', markersize=4.5)
fig, ax = plt.subplots()
# plt.plot(2*(784*hidden_layer_neurons+hidden_layer_neurons*10), np_accuracies[:,-1], '-.', color='blue')       # Dense NN FLOPs
# plt.plot(flops, np_accuracies[:,-1], '-.', color='orange')                           # Convolutional NN FLOPs
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
plt.plot(test_counter, test_losses, **marker_style, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# ax.set_xscale('log')
# plt.xlabel('FLOPs')
# plt.ylabel('Accuracy (%)')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.savefig(os.path.join(projects_directory_location,f'epochs\\trainingcurve.png'),
            bbox_inches='tight')

print('Reached end of code. Terminating program.')