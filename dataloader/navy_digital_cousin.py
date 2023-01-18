import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Libraries for generating confusion matrix
from sklearn.metrics import confusion_matrix
#import seaborn as sn
import pandas as pd

# Duplicate stdout to text file
class Logger(object):
    def __init__(self, should_print=True):
        self.terminal = sys.stdout
        self.should_print = should_print
        self.log = open("logfile.txt", "a")
        self.log.write( "---Log for Machine Learning Exploration---\n" )
    
    def write_always(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def write(self, message):
        if self.should_print:
            self.terminal.write(message)
            self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    
sys.stdout = Logger(should_print=False)
root = os.getcwd()#+"/.."

##### DEFINE PARAMETERS #####
# General Params
log_interval = 300

# Training Dataset and Optimization Parameters
image_size = 24     #px
batch_size_train = 64
batch_size_evaluation = 1000
learning_rate = 0.001
momentum = 0.5
n_classes = 10

# Training args
dataset_options = ["MNIST", "EMNIST", "CIFAR10"]
optim_options = ["SGD", "Adam"]
parser = argparse.ArgumentParser( description = "Try a digital cousin experiment" )
parser.add_argument( '-e', '--epochs', type=int, help="Number of epochs to train the network (should be >=1); default is 50.", default=50 )
parser.add_argument( '-r', '--random_seed', type=int, help="Random seed; default is 2.", default=2)
parser.add_argument( '-d', '--dataset', type=str, help="Dataset (MNIST, EMNIST, etc.); default is mnist.", choices=dataset_options, default=dataset_options[0])
parser.add_argument( '-c', '--cf_mat', type=bool, help="Create confusion matrix; default is false", default=False)
parser.add_argument( '-o', '--optim', type=str, help="Optimization algorithm (SGD, Adam, etc.); default is Adam.", choices=optim_options, default="Adam")
parser.add_argument( '-z', '--hpt', help="Hyperparameter tuning; default is false", default=False, action='store_true')
# Architecture args
layer_options = ["CONV", "LINEAR"]
activation_options = ["RELU", "LINEAR"]
parser.add_argument( '-l', '--layer_type', type=str, help="Layer type (CONV or LINEAR); default is conv. Inputs are flattened if not linear selected.", choices=layer_options, default="conv")
# conv
parser.add_argument( '-p', '--dropout', type=float, help="Probablity of dropout (should be between 0 and 1 inclusive); default is 0.5.", default=0.5)
parser.add_argument( '-f', '--num_filters', type=int, help="Number of convolutional filters. default is 9.", default=9)
parser.add_argument( '-k', '--kernel_size', type=int, help="Square dim of conv kernel; default is 3.", default=3)
parser.add_argument( '-q', '--no_fc_conv_output', dest='fc_conv_output',action='store_false', help="Use a second convolutional layer instead of dense layers for output.", default=True)
parser.add_argument( '-y', '--use_single_conv', dest='single_conv',action='store_true', help="Use a single convolutional layer.", default=False)
# linear
parser.add_argument( '-w', '--layer_width', type=int, help="Number of hidden units per layer; default is 4096.", default=4096) 
parser.add_argument( '-g', '--hidden_depth', type=int, help="Number of hidden dense layers; default is 1.", default=1) #TODO add support for this
parser.add_argument( '-a', '--activation', type=str, help="Activation function (RELU or LINEAR) default is relu.", default="RELU", choices=activation_options)
# both
parser.add_argument( '-s', '--switch_test_activation', help="If activation should be switched for training and testing. Default is false.", default=False, action='store_true')

args = parser.parse_args()

# Arg Validation
if args.epochs <= 0:
    print("Number of epochs must be greater than 0. Defaulting to 50.")
    args.epochs = 50

if args.dropout <= 0 or args.dropout >= 1: 
    print("Dropout probability must be between 0 and 1. Defaulting to 0.5.")
    args.dropout = 0.5

if args.num_filters <= 0:
    print("Number of convolutional channels must be greater than 0. Defaulting to 9.")
    args.num_filters = 9
    
if args.kernel_size <= 0:
    print("Kernel size must be greater than 0. Defaulting to 1.")
    args.kernel_size = 1

if args.layer_width <= 0:
    print("Layer size must be greater than 0. Defaulting to 1.")
    args.layer_width =9
    
if args.hidden_depth <= 0:
    print("Number of hidden layers must be greater than 0. Defaulting to 9.")
    args.hidden_depth = 1

##### END PARAMETERS #####

##### SETUP #####
print(root)
# Output & Save Paths and Validation
RESULTS_PATH = os.path.join(root,f'results_{args.dataset}_{args.layer_type}_base')
MODEL_PATH = os.path.join(root,f'results_{args.dataset}_{args.layer_type}_base/model.pth')
OPTIMIZER_PATH = os.path.join(root,f'results_{args.dataset}_{args.layer_type}_base/optimizer.pth')
DATA_PATH = os.path.join(root,'data',f'data_{args.dataset}') # @ todo move to ../../data, etc.
print(DATA_PATH)

if not os.path.isdir(RESULTS_PATH): os.mkdir(RESULTS_PATH)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# PyTorch Backend Parameters
#torch.backends.cudnn.enabled = False        # Disables nondeterministic algorithms for cuDNN
torch.manual_seed(args.random_seed)

##### END SETUP #####

##### DATASET LOADING #####
# 0.1307 and 0.3081 are the mean and stdev for the MNIST dataset respectively.
transforms_base =  [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
if ( args.dataset == "MNIST" ):
    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=torchvision.transforms.Compose(transforms_base)),
            batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(DATA_PATH, train=False, download=True,transform=torchvision.transforms.Compose(transforms_base)),
            batch_size=batch_size_evaluation, shuffle=True)
if ( args.dataset == "EMNIST" ):
    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.EMNIST(DATA_PATH, split="balanced", train=True, download=True,transform=torchvision.transforms.Compose(transforms_base),
            batch_size=batch_size_train, shuffle=True))
    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.EMNIST(DATA_PATH, split="balanced", train=False, download=True,transform=torchvision.transforms.Compose(transforms_base)),    
            batch_size=batch_size_evaluation, shuffle=True)
if ( args.dataset == "CIFAR10" ):
    transforms_base.insert(0, torchvision.transforms.Grayscale(num_output_channels = 1))    # convert RGB, https://stackoverflow.com/questions/52439364/how-to-convert-rgb-images-to-grayscale-in-pytorch-dataloader
    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=torchvision.transforms.Compose(transforms_base)),
            batch_size=batch_size_train, shuffle=True)    
    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(DATA_PATH, train=False, download=True,transform=torchvision.transforms.Compose(transforms_base)),
            batch_size=batch_size_evaluation, shuffle=True)    
##### END DATASET LOADING #####

##### MODEL DEF #####
class Conv_Net(nn.Module):
    def __init__(self,num_classes=10, activation="RELU"):
        super(Conv_Net, self).__init__()
        conv_output = int((image_size - 1*(args.kernel_size-1) - 1)/args.kernel_size + 1)
        pooled_output = int((conv_output - 1*(2-1) - 1)/2 + 1)
        self.flattened_width = int(args.num_filters*pooled_output**2)
        self.fc_conv_output = args.fc_conv_output
        self.activation = activation
        self.resize = torchvision.transforms.Resize(size=(image_size, image_size))
        
        self.single_conv = args.single_conv
        if self.single_conv:
            self.conv1 = nn.Conv2d(1, num_classes, kernel_size=args.kernel_size, stride=args.kernel_size)
            return 
        else: 
            self.conv1 = nn.Conv2d(1, args.num_filters, kernel_size=args.kernel_size, stride=args.kernel_size)
        if self.fc_conv_output:
            self.flatten = nn.Flatten()
            self.linear1 = nn.Linear(self.flattened_width, 35)
            self.linear2 = nn.Linear(35, num_classes)
        else:
            self.conv2 = nn.Conv2d(args.num_filters, num_classes, kernel_size=args.kernel_size, stride=args.kernel_size)
        
    def forward(self, x):
        x = self.resize(x)
        x = self.conv1(x)
        if self.single_conv:
            x = F.max_pool2d(x, x.shape[-1]) # Down to num_classes x 1 x 1
            x = x.view(-1, x.shape[1]) # Down to num_classes
            return F.log_softmax(x)
        if self.fc_conv_output:
            x = F.dropout2d(x, p=args.dropout) if args.dropout > 0 else x
            x = F.max_pool2d(x,2)
            x = F.relu(x) if self.activation == "RELU" else x
            
            x = self.flatten(x)
            
            x = self.linear1(x)
            x = F.relu(x) if self.activation == "RELU" else x
            x = self.linear2(x)
        else:
            x = F.dropout2d(x, p=args.dropout) if args.dropout > 0 else x
            x = F.max_pool2d(x, 2) # Down to num_classes x 1 x 1
            x = F.relu(x) if self.activation == "RELU" else x
            x = self.conv2(x)
            x = F.max_pool2d(x, x.shape[-1]) # Down to num_classes x 1 x 1
            x = F.relu(x) if self.activation == "RELU" else x
            x = x.view(-1, x.shape[1]) # Down to num_classes
        
        return F.log_softmax(x)

class Linear_Net(nn.Module):
    def __init__(self,num_classes=10, activation="RELU"):
        super(Linear_Net, self).__init__()
        self.resize = torchvision.transforms.Resize(size=(image_size, image_size))
        self.flatten = nn.Flatten()
        self.activation = activation
        self.depth = args.hidden_depth
         
        self.core_layers = nn.ModuleList()
        if self.depth == 1: self.core_layers.append(nn.Linear(image_size*image_size, num_classes))
        else:
            self.core_layers.append(nn.Linear(image_size*image_size, args.layer_width))            
            for i in range(args.hidden_depth-2):
                self.core_layers.append(nn.Linear(args.layer_width, args.layer_width))
            self.core_layers.append(nn.Linear(args.layer_width, num_classes))

    def forward(self, x):
        x = self.resize(x)
        x = self.flatten(x)
        x = self.core_layers[0](x)
        if self.depth > 1:
            x = F.relu(x) if self.activation == "RELU" else x
            for i in self.core_layers[1:-1]:
                x = i(x)
                x = F.relu(x) if self.activation == "RELU" else x
            x = self.core_layers[-1](x)
        
        return F.log_softmax(x)

def swapped_activation_model(model, num_classes=10):
    """ Builds identical model and copies weights but swaps linear / non linear activation functions for each layer. """
    # Swap activation
    if args.activation == "RELU": new_activation = "LINEAR"
    else: new_activation = "RELU"
    
    # Rebuild model
    predictor_network = Linear_Net(num_classes=num_classes, activation=new_activation) if args.layer_type == 'LINEAR' else Conv_Net(num_classes=num_classes, activation=new_activation)
    
    predictor_network.eval()
    
    return predictor_network
    
def copy_weights(model, predictor_network):
    # Copy weights from model to predictor
    if args.switch_test_activation: predictor_network.load_state_dict(model.state_dict())
    return predictor_network
##### END MODEL DEF #####

##### TRAINING & evaluation#####
def train(epoch, optimizer, network):
    network.train()
    epoch_train_loss = 0
    train_counter = []
    for batch_idx, (data, target) in enumerate(train_loader):
        # Batch idx is batch num in the training set; data is batch training data; target is batch labels
        optimizer.zero_grad() # Zero out gradients
        output = network(data) # Forward pass
        
        # Get the negative log-likelihood loss between output and ground truth.
        loss = F.nll_loss(output, target)
        # Collect the gradients.
        loss.backward()
        optimizer.step()
        # Collect loss
        epoch_train_loss += loss.item()
        
        # Logging & Saving
        if batch_idx % log_interval == 0:
            pct_epoch = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} \tEp % Complete: {pct_epoch:.0f}%\tLoss: {loss.item():.6f}')
            
            # Create lists for saving training and evaluation losses
            train_counter.append((batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
            
            # Save NN and optimize states
            try:
                torch.save(network.state_dict(), MODEL_PATH)
                torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
            except Exception as ex:  print("Error saving model: ", ex)

    train_loss_normed = epoch_train_loss / int(len(train_loader.dataset)/batch_size_train)
    return train_counter, train_loss_normed

def evaluation(model, swapped_activation=False):
    # Set model to evaluation mode
    model.eval()
    evaluation_loss, num_correct = 0, 0
    
    # Disable gradient calculation
    with torch.no_grad():        
        for data, target in test_loader:
            # Forward
            output = model(data)
            # Collect loss
            evaluation_loss += F.nll_loss(output, target, size_average=False).item()
            # Get predictions
            pred = output.data.max(1, keepdim=True)[1]
            # Count correct predictions
            num_correct += pred.eq(target.data.view_as(pred)).sum()
    
    # Normalize evaluation loss
    evaluation_loss_normed = evaluation_loss / len(test_loader.dataset)
    accuracy = (float(100. * num_correct / len(test_loader.dataset)))
    
    print(f'Evaluation set{" [swapped]" if swapped_activation else "          "}: Avg. loss: {evaluation_loss_normed:.4f}, Accuracy: {num_correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')
    return evaluation_loss_normed, accuracy
    
def gen_confusion_matrix():
    y_pred, y_true = [], []
    # iterate over test data
    for inputs, labels in test_loader:
        # Feed Network
        output = network(inputs)
        # Get predictions
        y_pred.extend((torch.max(torch.exp(output), 1)[1]).data.cpu().numpy())
        # Get labels
        y_true.extend(labels.data.cpu().numpy())

    # constant for classes
    if(dataset == "MNIST"): classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    elif(dataset == "CIFAR10"): classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    elif(dataset == "EMNIST"):
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 
                   'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                   'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
                   'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't')

    # Build & Save confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_matrix_df = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *n_classes, index = [i for i in classes],
                         columns = [i for i in classes])
    plt.figure(figsize = (15,8))
    ax = sn.heatmap(cf_matrix_df, annot = True, linewidth = 0.95, xticklabels = True, yticklabels = True)
    ax.set(xlabel="Predicted", ylabel="True")
    ax.tick_params(axis='y', rotation=90)
    ax.tick_params(bottom=True, top=False, left=True, right=False)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    plt.savefig(RESULTS_PATH + '/heatmap_{}.png'.format(dataset))
##### END TRAINING & evaluation #####

##### MAIN #####s
# Initialize network and optimizer.
def normal_main():
    num_classes = 10
    network = Linear_Net(num_classes=num_classes, activation=args.activation) if args.layer_type == 'LINEAR' else Conv_Net(num_classes=num_classes, activation=args.activation)
    eval_network = swapped_activation_model(network, num_classes=num_classes) if args.switch_test_activation else network
    optimizer = optim.Adam(network.parameters(), lr=learning_rate) if args.optim == 'Adam' else optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)

    print(f"Using model: {args.layer_type}")
    if args.switch_test_activation: print(f"Training Activation: {args.activation}\tTesting Activation: {args.activation if not args.switch_test_activation else 'opposite'}")
    print(f"Using optimizer: {args.optim}")

    print(network)
    print(eval_network)

    train_losses_normed, train_counter = [], []
    evaluation_losses_normed = []
    evaluation_losses_normed_swapped = []
    accuracies = []
    accuracies_swapped = []

    # Perform training loop
    copy_weights(network, eval_network)
    if args.switch_test_activation: 
        evaluation_loss_normed, accuracy = evaluation(eval_network, swapped_activation=True)
        evaluation_losses_normed_swapped.append(evaluation_loss_normed)
        accuracies_swapped.append(accuracy)
        
    evaluation_loss_normed, accuracy = evaluation(network, swapped_activation=False)
    evaluation_losses_normed.append(evaluation_loss_normed)
    accuracies.append(accuracy)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_counter_, train_loss_normed = train(epoch, optimizer, network)
        train_losses_normed.append(train_loss_normed)
        train_counter.extend(train_counter_)
        
        # Evaluate
        copy_weights(network, eval_network)
        if args.switch_test_activation: 
            evaluation_loss_normed, accuracy = evaluation(eval_network, swapped_activation=True)
            evaluation_losses_normed_swapped.append(evaluation_loss_normed)
            accuracies_swapped.append(accuracy)
            
        evaluation_loss_normed, accuracy = evaluation(network, swapped_activation=False)
        evaluation_losses_normed.append(evaluation_loss_normed)
        accuracies.append(accuracy)


    # Plot training curve
    fig, ax = plt.subplots()
    plt.plot(range(1,args.epochs+1), train_losses_normed, color='blue', label='Train Loss')
    plt.scatter(range(0,args.epochs+1), evaluation_losses_normed, color='red', label='evaluation Loss')
    plt.legend(loc='upper right')
    #ax.set_xscale('log')
    #plt.ylim(1.0, 2.1)
    #plt.xlabel('number of training examples seen')
    plt.xlabel('Epoch #')
    plt.ylabel('negative log likelihood loss')
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + '/performancediagram.png',
                bbox_inches='tight')

    if(args.cf_mat == True): gen_confusion_matrix()
        
    #print('End.\nSee files in {}.'.format( RESULTS_PATH ))
    return accuracies_swapped[-1], accuracies[-1] # First is swapped, second is not swapped

def HPT_main():
    import tqdm, itertools
    
    args.epochs = 5
    
    #layer_types = ["CONV"]
    #dropouts = [0.5]
    #num_filters = [10, 20, 40, 80, 120]
    #kernel_sizes = [3]
    #fc_conv_outputs = [False, True]
    #layer_widths = [4096]
    #hidden_depths = [1]
    #activations = ["RELU", "LINEAR"]
    #switch_test_activations = [True]
    
    layer_types = ["LINEAR"]
    dropouts = [0.2,0.5,0.7]
    num_filters = [10]
    kernel_sizes = [3]
    fc_conv_outputs = [True]
    layer_widths = [50,500,1024,2048,4096]
    hidden_depths = [1,2,3,4]
    activations = ["RELU", "LINEAR"]
    switch_test_activations = [True]
    
    options = [layer_types, dropouts, num_filters, kernel_sizes, fc_conv_outputs, layer_widths, hidden_depths, activations, switch_test_activations]
    combos = [p for p in itertools.product(*options)]
    
    for combo in tqdm.tqdm(combos, desc="HPT Combos:"): 
        args.layer_type = combo[0]
        args.dropout = combo[1]
        args.num_filters = combo[2]
        args.kernel_size = combo[3]
        args.fc_conv_output = combo[4]
        args.layer_width = combo[5]
        args.hidden_depth = combo[6]
        args.activation = combo[7]
        args.switch_test_activation = combo[8]
        res = normal_main()
        sys.stdout.write_always("****_START_****")
        sys.stdout.write_always(f"layer_type: {combo[0]}\tdropout: {combo[1]}\tnum_filters: {combo[2]}\tkernel_size: {combo[3]}\tfc_conv_output: {combo[4]}\tlayer_width: {combo[5]}\thidden_depth: {combo[6]}\tactivation: {combo[7]}\tswitch_test_activation: {combo[8]}")
        sys.stdout.write_always(f"Accuracy [swapped]: {res[0]}, \t Accuracy [not swapped]: {res[1]}")
        sys.stdout.write_always("****__END__****")
        
        
    

if __name__ == '__main__':
    if args.hpt == True: HPT_main()
    else: normal_main()