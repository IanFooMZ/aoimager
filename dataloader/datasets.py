import sys
import os
import numpy as npf
import autograd.numpy as np
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from skimage import transform as skt
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

def get_processed_dataset(train_bool, name, img_dimension, process_args=None):
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

def augment(training_set, training_set_transform, augmentations):
	'''Performs image augmentation on the given sample (type Subset)
 	Input: training samples [Subset], targets, and the types of augmentations that are to be performed.
	Output: augmented (larger) Dataset, and corresponding targets.
 	'''
  
	#* Augmentation functions are taken from the following references:
	# https://towardsdatascience.com/augmenting-images-for-deep-learning-3f1ea92a891c
	# https://towardsdatascience.com/a-comprehensive-guide-to-image-augmentation-using-pytorch-fb162f2444be
	# https://www.analyticsvidhya.com/blog/2019/12/image-augmentation-deep-learning-pytorch/#h2_5
	# https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#gaussianblur
 
	augmented_set = []
	augmented_set_targets = []
 
	for set_idx in range(len(training_set)):
		img = training_set[set_idx][0]
		tgt = training_set[set_idx][1]
		augmented_set.append(img)
		augmented_set_targets.append(tgt)

		# Append an augmented copy of the image with some probability p
		prob_edit_img = 0.3
		if np.random.uniform() <= prob_edit_img and augmentations:
			# Loop through each augmentation in the augmentation array defined in config, applying it in turn to the image.
			for agm in augmentations:
				# NOTE: img.size() = [1, n_x, n_y]. Maintain this for the output.
		
				if agm['type'] in ['rotation']:
					if agm['wrap']:		# Setting mode as ‘wrap’ fills the points outside the boundaries of the input with the remaining pixels of the image.
						rotate_angle = agm['angle_range']
						if agm['random']:	
							rotate_angle*= np.random.uniform(-1,1)
						img[0] = torch.from_numpy(skt.rotate(img[0].detach().numpy(), angle=rotate_angle, mode='wrap'))
					else:				# No wrap means just a straightforward rotation and the remainder is filled in with a uniform color.
						if agm['random']:
							img = torchvision.transforms.RandomRotation(degrees=agm['angle_range'])(img)
						else:	img = torchvision.transforms.functional.rotate(img, angle=agm['angle_range'])
		
				elif agm['type'] in ['shift']:
					x_shift = agm['x'] * img.size()[1]
					y_shift = agm['y'] * img.size()[2]
					if agm['random']:
						x_shift *= np.random.uniform(-1,1)
						y_shift *= np.random.uniform(-1,1)
					x_shift = np.floor(x_shift).astype(int)
					y_shift = np.floor(y_shift).astype(int)
					transform = skt.AffineTransform(translation=(x_shift,y_shift))
					img[0] = torch.from_numpy(skt.warp(img[0].detach().numpy(), transform, mode='wrap'))

				elif agm['type'] in ['flip']:
					if agm['lr']:
						img[0] = torch.from_numpy(np.fliplr(img[0].detach().numpy()).copy())
					if agm['ud']:
						img[0] = torch.from_numpy(np.flipud(img[0].detach().numpy()).copy())

				elif agm['type'] in ['intensity']:
					if agm['random']:
						img = torchvision.transforms.ColorJitter(brightness=agm['brightness_factor'])(img)
					else:
						from skimage.exposure import rescale_intensity
						min_output_intensity = torch.min(img[0]).item() * (1+agm['brightness_factor'])
						max_output_intensity = torch.max(img[0]).item() * (1+agm['brightness_factor'])
						img[0] = torch.from_numpy(rescale_intensity(img[0].detach().numpy(), in_range='image', 
														out_range=(min_output_intensity, max_output_intensity)))

				elif agm['type'] in ['random_noise']:
					from skimage.util import random_noise
					img[0] = torch.from_numpy(random_noise(img[0].detach().numpy(), var=agm['sigma']**2))
		
				elif agm['type'] in ['gaussian_blur']:
					if not agm['multichannel']:
						img = torchvision.transforms.GaussianBlur(kernel_size=(3,3), sigma=agm['sigma'])(img)
					else:	
						from skimage.filters import gaussian
						# This would ensure each channel was filtered separately
						img[0] = torch.from_numpy(gaussian(img[0].detach().numpy(),sigma=1,multichannel=True))	
		
				elif agm['type'] in ['random_crop']:
					img = torchvision.transforms.RandomResizedCrop(size=np.shape(img[0].detach().numpy()))(img)
					# If need from center - use torchvision.transforms.functional.resized_crop()

				elif agm['type'] in ['random_blocks']:
					def add_random_boxes(img, num, size, mode='L'):
						# See https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes for mode explanation
						img = np.asarray(img)
						img_size = img.shape[0]
						h,w = np.floor(size*img_size).astype(int), np.floor(size*img_size).astype(int)
						boxes = []
						for k in range(num):
							y,x = np.random.randint(0,img_size-w,(2,))
							img[y:y+h,x:x+w] = 0
							boxes.append((x,y,h,w))
						# img = Image.fromarray(img.astype('uint8'), mode=mode)
						return img

					img[0] = torch.from_numpy(add_random_boxes(img[0].detach().numpy(), agm['number'], agm['size']))

			# plt.subplot(1,2,1)
			# plt.imshow(np.squeeze(training_set[set_idx][0]))
			# # plt.colorbar()
			# plt.subplot(1,2,2)
			# plt.imshow(np.squeeze(img))
			# # plt.colorbar()
			# plt.show()
		
			augmented_set.append(img)
			augmented_set_targets.append(tgt)

	# Processing to get it into the right shape
	augmented_set = torch.from_numpy(np.squeeze(np.array(augmented_set)))
	augmented_set_targets = torch.from_numpy(np.array(augmented_set_targets)).type(torch.LongTensor)
	print(f'Training set was {len(training_set)} long; Augmented set is {len(augmented_set)} long.')
	
	# Shuffle
	# TODO: If you wrap the dataset in main.py with a DataLoader, it can do this part for you!
	idx_list = torch.randperm(augmented_set_targets.nelement())
	augmented_set_targets = augmented_set_targets.view(-1)[idx_list].view(augmented_set_targets.size())
	augmented_set = augmented_set[idx_list].view(augmented_set.size())
	
	processed_dataset = CustomDataset(augmented_set, augmented_set_targets, transform=training_set_transform)
	return processed_dataset, augmented_set_targets

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
	try:
		data_zeros = torch.zeros([len(regular_dataset), regular_dataset[0][0].size(1), regular_dataset[0][0].size(2)])
	except Exception as err:		# Probably means it's not a Subset - get the dimensions from the back instead
		data_zeros = torch.zeros([len(regular_dataset), regular_dataset[0][0].size(-2), regular_dataset[0][0].size(-1)])
	
	data_transformed = torch.complex(data_zeros, data_zeros)
	# targets = regular_dataset.targets
		  
	
	print(f'Fourier transforming original dataset to get planewaves.')
	# for dataset_idx in range(0,30):#len(regular_dataset)):
	for dataset_idx in range(0,len(regular_dataset)):
		if dataset_idx%(len(regular_dataset)//10) == 0:
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
