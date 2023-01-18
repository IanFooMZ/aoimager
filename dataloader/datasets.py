import sys
import os
import numpy as npf
import autograd.numpy as np
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from collections import Counter
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *

class CustomDataset(torch.utils.data.Dataset):
	def __init__(self, data, targets, transform=None):
		self.data = data
		self.targets = torch.LongTensor(targets)
		self.transform = transform
		
	def __getitem__(self, index):
		x = self.data[index]
		y = self.targets[index]
		
		# # We are going to overwrite this section since all transforms should have been already performed 
  		# # when building this custom dataset.
		# if self.transform:
		# 	x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
		# 	x = self.transform(x)
		
		return x, y
	
	def __len__(self):
		return len(self.data)

def get_processed_dataset(train_bool, name, img_dimension):
	'''This function contains several different instructions for loading, splitting and pre-processing datasets depending on the input arguments.
	Output: Processed dataset'''

	processed_dataset = None

	if name.upper() in ['MNIST']:
		# 0.1307 and 0.3081 are the mean and stdev for the MNIST dataset respectively.
		processed_dataset = torchvision.datasets.MNIST('/files/', train=train_bool, download=True,
											transform=torchvision.transforms.Compose([
											torchvision.transforms.Resize(img_dimension),
											torchvision.transforms.CenterCrop(img_dimension),
											torchvision.transforms.ToTensor(),
											torchvision.transforms.Normalize(
												(0.1307,), (0.3081,))
											]))
		
	return processed_dataset

def fourier_transform_dataset_with_NA(regular_dataset, targets, transform,
									  kx_values, ky_values, 
									  frequencies, numerical_aperture):
	'''Workaround for not being able to apply a Fourier operation to each item of the original dataset directly.
	Creates a data_transformed array. For each item in the original dataset, pulls the data, Fourier transforms, applies NA mask.
	Finally, creates a new Dataset with the transformed data and the original targets.
	Input: Requires the targets and transform in order to create a new dataset - pass them explicitly because regular_dataset might be a Subset class.
	Output: Fourier-transformed CustomDataset.'''
	
	# Initialize transformed data and pull original targets.
	# data_zeros = torch.zeros([30, regular_dataset[0][0].size(1), regular_dataset[0][0].size(2)])
	data_zeros = torch.zeros([len(regular_dataset), regular_dataset[0][0].size(1), regular_dataset[0][0].size(2)])
	data_transformed = torch.complex(data_zeros, data_zeros)
	# targets = regular_dataset.targets
		  
	
	print(f'Fourier transforming original dataset to get planewaves.')
	# for dataset_idx in range(0,30):#len(regular_dataset)):
	for dataset_idx in range(0,len(regular_dataset)):
		if dataset_idx%(len(regular_dataset)/10)==0:
			print(f'Index: {dataset_idx}/{len(regular_dataset)}')
	 
		# Grab E(x,y) and FFT to get E(k)
		E_input = regular_dataset[ dataset_idx ][0]			# When accessing an index of a dataset, [0] is the image, [1] is the groundtruth
		# Ek_input_original = np.fft.fftshift( np.fft.fft2( E_input ) )
		Ek_input = utility.torch_2dft(E_input)			
  		#! This is not the same as Greg's method in this part - there is an extra ifftshift in the torch_2dft() method which should be correct.
  		#! It may turn up errors in the future. Always check and recheck
  
		Ek_input = utility.apply_NA_in_kspace(Ek_input,
										kx_values, ky_values, frequencies, numerical_aperture)

		# Store value in E(k) dataset
		data_transformed[ dataset_idx ] = Ek_input
	
	# # Quick printing code to check that data_transformed is the FT of regular_dataset
	# y = CustomDataset(data_transformed, targets, transform=regular_dataset.transform)
	# fig,axs = plt.subplots(2)
	# axs[0].imshow(regular_dataset[0][0][0])
	# axs[1].imshow(np.real(utility.torch_2dift(y[0][0])))
	# plt.show()
 
	return CustomDataset(data_transformed, targets, transform=transform)



class EditableMNIST(torchvision.datasets.MNIST):
	'''An attempt to inherit torchvision.datasets.MNIST to override the __getitem__() function to return an editable list instead of an uneditable tuple.
	The purpose was to directly replace the images stored in the dataset with Fourier-processed images.
	! This did not work ! -20230109, Ian'''
	def __init__(self,
			root: str,
			train: bool = True,
			transform: Optional[Callable] = None,
			target_transform: Optional[Callable] = None,
			download: bool = False
		) -> None:
		super(torchvision.datasets.MNIST).__init__(root, train=train, transform=transform, 
						 target_transform=target_transform, download=download)
	
	def __getitem__(self, index: int) -> List[Union[Any, Any]]:
		"""
		Args:
			index (int): Index

		Returns:
			List: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], int(self.targets[index])

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img.numpy(), mode='L')

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self) -> int:
		return len(self.data)
