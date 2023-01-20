import sys
import os
import copy
import numpy as npf
import autograd.numpy as np
import torch
from datetime import datetime

class Logger(object):
	def __init__(self, data_folder):
		self.terminal = sys.stdout
		self.log = open(os.path.join(data_folder,"logfile.txt"), "a")
		
		self.log.write( "---Log: Optical Neural Network---\n" )
		self.log.write( "Time Started: " + datetime.now().strftime(r"%d/%m/%Y %H:%M:%S") + "\n")
   
	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)  

	def flush(self):
		# this flush method is needed for python 3 compatibility.
		# this handles the flush command by doing nothing.
		# you might want to specify some extra behavior here.
		pass    

def torch_2dft(input):
	ft = torch.fft.ifftshift(input)
	ft = torch.fft.fft2(ft)
	return torch.fft.fftshift(ft)

def torch_2dift(input):
	ift = torch.fft.ifftshift(input)
	ift = torch.fft.ifft2(ift)
	ift = torch.fft.fftshift(ift)
	return ift

def calculate_2dft(input):
	ft = npf.fft.ifftshift(input)
	ft = npf.fft.fft2(ft)
	return npf.fft.fftshift(ft)

def calculate_2dift(input):
	ift = npf.fft.ifftshift(input)
	ift = npf.fft.ifft2(ift)
	ift = npf.fft.fftshift(ift)
	return ift.real

def apply_NA_in_kspace(image_fft, kx_values, ky_values, frequencies, numerical_aperture):
	'''Takes as input an FFT'd 2D image and then applies NA mask based on kr_max which is given by the ranges of theta and phi.
	Output: 2D image in Fourier space with k-mask'''
 
	image_fft_xsize = np.shape(image_fft)[-2]
	image_fft_ysize = np.shape(image_fft)[-1]
	
	num_kx = len(kx_values)
	num_ky = len(ky_values)
	kx_start = ( image_fft_xsize // 2 ) - ( num_kx // 2 )
	ky_start = ( image_fft_ysize // 2 ) - ( num_ky // 2 )
	if image_fft_xsize != image_fft_ysize:
		print('Alert! The input image is not square.')
		sys.exit(1)
 
	# Apply NA mask based on kr_max which is given by the ranges of theta and phi
	for kx_idx in range( 0, num_kx ):
		for ky_idx in range( 0, num_ky ):

			kx = kx_values[ kx_idx ]
			ky = ky_values[ ky_idx ]
			

			kr = np.sqrt( kx**2 + ky**2 )
			kr_max = ( 2 * np.pi * frequencies[ 0 ] ) * numerical_aperture

			if kr >= kr_max:
				try:		# assuming image_fft is 3-D
					image_fft[0, kx_idx + kx_start, ky_idx + ky_start ] = 0
				except Exception as err:	# otherwise, it's 2-D
					image_fft[ kx_idx + kx_start, ky_idx + ky_start ] = 0
	
	return image_fft

def define_circular_zones_around_center(num_classes, x_dim, y_dim, central_radius, zone_radius):
	'''Defines zones according to http://jsfiddle.net/coma/nk0ms9hb/.
	Generates <num_classes> zones, each of radius <zone_radius>, arranged around a central dark zone of radius <central_radius>.
	Output: List of len <num_classes> of images with one-hot zones i.e. output[0] is an image with the zone for class 0 lit up and so on.'''
	
	# Testing image to visualize all zones at once
	array = torch.zeros(x_dim, y_dim)
	# List of images each corresponding to one class with the selected zone lit up (1.0) and all other values set to 0
	onehot_arrays = torch.zeros(num_classes, x_dim, y_dim)
 
	# Main circle:
	X_shift = x_dim//2
	Y_shift = y_dim//2
	x_vals = torch.linspace(-X_shift, X_shift, steps=x_dim)
	y_vals = torch.linspace(-Y_shift, Y_shift, steps=y_dim)
 
	def draw_circle(array, x0,y0, radius, intensity):
		for x_idx, x_val in enumerate(x_vals):
			for y_idx, y_val in enumerate(y_vals):
				if (x_val - x0)**2 + (y_val - y0)**2 <= radius**2:
					array[x_idx, y_idx] += intensity

		return copy.deepcopy(array)
	 
	array = draw_circle(array, 0,0, central_radius, 1.0)
 
	# Algorithm to make the outer zones have tangentially touching edges
	artificially_enlarged_zone_radius = zone_radius*10/9
	cos_dtheta = torch.FloatTensor([1 - 2*artificially_enlarged_zone_radius**2/(central_radius+artificially_enlarged_zone_radius)**2])
	d_theta = torch.acos(cos_dtheta).item()
	
	# if 2*torch.pi/num_classes < d_theta:
	# 	print('Detection Plane: Zones overlapping! Either increase the central radius, or decrease the outer zone radii!')
	# 	sys.exit(1)
	
	# Override this with equal angular spacing around the central circle
	d_theta = torch.max(torch.FloatTensor([d_theta, 2*torch.pi/num_classes]))



	angles = torch.arange(0, num_classes*d_theta, d_theta)
	for ang_idx, angle in enumerate(angles):
		array = draw_circle(array,
							(central_radius+zone_radius)*torch.cos(angle), 
							(central_radius+zone_radius)*torch.sin(angle),
							zone_radius, 50.0)
		onehot_arrays[ang_idx] = draw_circle(torch.zeros(x_dim, y_dim),
											(central_radius+zone_radius)*torch.cos(angle), 
											(central_radius+zone_radius)*torch.sin(angle),
											zone_radius, 1.0)
	
	# # Visualization code for Testing
	# import matplotlib.pyplot as plt
	# plt.imshow(array)
	# plt.colorbar()
	# plt.show()
 
	return onehot_arrays

def target_classes_to_zones(ground_truth, img_dimension):
	'''Input: Tensor
	Output: Tensor'''
	
	# target_intensities = torch.zeros([len(ground_truth), img_dimension])
	
	# for target_idx, target in enumerate(ground_truth):
	# 	target_intensities[target_idx] = 3
 
	target_intensities = define_circular_zones_around_center(len(ground_truth),
														  img_dimension, img_dimension,
														  torch.ceil(torch.FloatTensor([img_dimension])*24/100), 
														  torch.ceil(torch.FloatTensor([img_dimension])*8/100)
														  )
	
	return target_intensities

def draw_circle_deprecated(shape,radius):
	'''
	Input:
	shape    : tuple (height, width)
	radius : scalar
	
	Output:
	npf.array of shape that says True within a circle of given radius, centered at the centerpoint of the image grid (shape) 
	'''
 
	# Make sure shape is 2D array
	assert len(shape) == 2
	
	TF = npf.zeros(shape,dtype=npf.bool)
	center = npf.array(TF.shape)/2.0

	for iy in range(shape[0]):
		for ix in range(shape[1]):
			TF[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < radius **2
   
   # 
	return(TF)

def kr_max( wavelength, img_mesh, img_dimension, numerical_aperture ):
	'''Calculates the kr_max (radius from center in k-space) based on a given numerical aperture.
	Inputs: wavelength (single value float), img_mesh (image mesh size), img_dimension (num. mesh points), numerical_aperture (float)
	Outputs: |k_max|, and a vector of k_radius values with |k|<|k_max|.'''

	# What is the maximum frequency present in the image? Theoretically this depends on the mesh space:
	kr_max_um_inv = 2 * npf.pi * ( img_dimension - 1 ) / ( img_dimension * img_mesh )
	# This is the width between positive and negative frequencies, so the corresponding kx, ky-grid must be:
	k_values_um_inv = npf.linspace( -0.5 * kr_max_um_inv, 0.5 * kr_max_um_inv, img_dimension )

	# Now impose the |k_max| that arises from our artificial NA i.e. the imaging system.
	kr_max = ( 2 * npf.pi / wavelength ) * numerical_aperture		# is it 4??? Δx = λ/2NA
	kr_max = ( 4 * npf.pi / wavelength ) * numerical_aperture		# is it 4??? Δx = λ/2NA
	#! kr_max_um_inv needs to be less than kr_max!
	# todo: calculate the relation between img_mesh and numerical_aperture to satisfy this condition

	kr_values = []
	mid_dimension = img_dimension // 2

	# Starting from -0.5*kr_max_um_inv, tick towards zero until you find the array index corresponding to |k_max| in the -ve region.
	start_kr_idx = 0
	while npf.abs( k_values_um_inv[ start_kr_idx ] ) > kr_max:
		start_kr_idx += 1

	# Starting from that array index, tick towards the end of the array until you find the index corresponding to |k_max| in the +ve region.
	end_kr_idx = start_kr_idx
	while npf.abs( k_values_um_inv[ end_kr_idx ] ) <= kr_max:
		end_kr_idx += 1

	# Truncate accordingly to grab only those k-coordinates corresponding to frequencies |k| < |k_max|.
	kr_values_cut = k_values_um_inv[ start_kr_idx : end_kr_idx ]
	# assert ( len( kr_values_cut ) % 2 ) == 1, 'Assert: Vector of kr values is cut to an odd length, for symmetry.'

	print(f'kr_max is {kr_max:.5f}; k-grid is {len(kr_values_cut)} wide in both dimensions.')
	return kr_max, kr_values_cut


def dynamic_weight_squared_proportion( figure_of_merit_individual_ ):
	flatten_fom = figure_of_merit_individual_.flatten()

	weights = flatten_fom**2 / np.sum( flatten_fom**2 )

	return np.reshape( weights, figure_of_merit_individual_.shape )


def Q_ramp_linear( Q_ramp_start_, Q_ramp_end_, Q_number_steps_ ):
	return np.linspace( Q_ramp_start_, Q_ramp_end_, Q_number_steps_ )

def Q_ramp_exponential( Q_ramp_start_, Q_ramp_end_, Q_number_steps_ ):
	x = np.linspace( np.log( Q_ramp_start_ ), np.log( Q_ramp_end_ ), Q_number_steps_ )

	return np.exp( x )

def c4_symmetry_explicit( permittivity, Nx, Ny, num_design_layers ):

	num_rotations = 4

	values = []
	for layer_idx in range( 0, num_design_layers ):

		get_layer = np.reshape( permittivity[ Nx * Ny * layer_idx : ( Nx * Ny * ( layer_idx + 1 ) ) ], [ Nx, Ny ] )
		get_layer = ( 1. / num_rotations ) * ( np.rot90( get_layer, k=0 ) + np.rot90( get_layer, k=1 ) + np.rot90( get_layer, k=2 ) + np.rot90( get_layer, k=3 ) )

		values.append( get_layer )

	eps_data = values[ 0 ].flatten()
	for idx in range( 1, len( values ) ):
		eps_data = np.concatenate( ( eps_data, values[ idx ].flatten() ) )

	return eps_data


def vertical_flip_symmetry_explicit( permittivity, Nx, Ny, num_design_layers ):

	values = []
	for layer_idx in range( 0, num_design_layers ):

		get_layer = np.reshape( permittivity[ Nx * Ny * layer_idx : ( Nx * Ny * ( layer_idx + 1 ) ) ], [ Nx, Ny ] )

		get_layer = ( 1. / 2 ) * ( get_layer + np.fliplr( get_layer ) )

		values.append( get_layer )

	eps_data = values[ 0 ].flatten()
	for idx in range( 1, len( values ) ):
		eps_data = np.concatenate( ( eps_data, values[ idx ].flatten() ) )

	return eps_data


def woodpile( permittivity, Nx, Ny, num_design_layers, start_axis=0 ):

	values = []

	dimensions = [ Nx, Ny ]

	for layer_idx in range( 0, num_design_layers ):

		get_layer = np.reshape( permittivity[ Nx * Ny * layer_idx : ( Nx * Ny * ( layer_idx + 1 ) ) ], [ Nx, Ny ] )

		average_axis = ( start_axis + layer_idx ) % 2

		get_layer = np.swapaxes(
				np.repeat(
					np.reshape( np.mean( get_layer, axis=average_axis ), [ 1, dimensions[ average_axis ] ] ),
					dimensions[ average_axis ],
					axis=0 ),
				0, average_axis )

		values.append( get_layer )

	eps_data = values[ 0 ].flatten()
	for idx in range( 1, len( values ) ):
		eps_data = np.concatenate( ( eps_data, values[ idx ].flatten() ) )

	return eps_data

def quadratic_optimization_goal(
	max_transmission,
	num_frequencies, num_polarizations, num_theta, num_phi, num_orders,
	polarizations, theta_values ):
	sin_theta = np.sin( theta_values )
	sin_sq_theta = sin_theta**2

	theta_transmission = max_transmission * sin_sq_theta / np.max( sin_sq_theta )

	transmission_goal = np.zeros( ( num_frequencies, num_polarizations, num_theta, num_phi, 2, num_orders ), dtype=np.complex128 )

	for theta_idx in range( 0, num_theta ):
		if ( len( polarizations ) == 1 ) and ( polarizations[ 0 ] == 'p' ):
			transmission_goal[ :, 0, theta_idx, :, 1, 0 ] = theta_transmission[ theta_idx ]
		else:
			for input_polarization_idx in range( 0, num_polarizations ):
				transmission_goal[ :, input_polarization_idx, theta_idx, :, input_polarization_idx, 0 ] = theta_transmission[ theta_idx ]

	return transmission_goal



def phase_imaging_shift_optimization_goal(
	max_transmission, shift_amount,
	num_frequencies, num_polarizations, num_theta, num_phi, num_orders,
	frequencies, polarizations, theta_values, phi_values, orders=None ):

	wavelengths = 1. / frequencies

	# assert num_orders == 1, 'For now, we are just doing the zeroth order!'

	transmission_goal = np.zeros( ( num_frequencies, num_polarizations, num_theta, num_phi, 2, num_orders ), dtype=np.complex )

	phase_by_order = np.zeros( num_orders )
	amplitude_by_order = np.zeros( num_orders )

	if orders is None:
		orders = [ 0, 1, 4 ]

	assert len( orders ) == 3, 'Currently operating with three imaged orders'

	phase_by_order[ orders[ 0 ] ] = 0
	phase_by_order[ orders[ 1 ] ] = 2 * np.pi / 3.
	phase_by_order[ orders[ 2 ] ] = 4 * np.pi / 3.

	amplitude_by_order[ orders[  0 ] ] = 1. / 3.
	amplitude_by_order[ orders[ 1 ] ] = 1. / 3.
	amplitude_by_order[ orders[ 2 ] ] = 1. / 3.

	for frequency_idx in range( 0, num_frequencies ):

		for phi_idx in range( 0, num_phi ):
			get_phi = phi_values[ phi_idx ]

			kx = 2 * np.pi * np.sin( theta_values ) * np.cos( get_phi ) / wavelengths[ frequency_idx ]

			for order_idx in range( 0, num_orders ):

				theta_transmission = max_transmission * 0.5 * amplitude_by_order[ order_idx ] * ( 1 + np.exp( 1j * phase_by_order[ order_idx ] ) * np.exp( -1j * kx * shift_amount ) )

				for theta_idx in range( 0, num_theta ):
					if ( len( polarizations ) == 1 ) and ( polarizations[ 0 ] == 'p' ):
						transmission_goal[ frequency_idx, 0, theta_idx, phi_idx, 1, order_idx ] = theta_transmission[ theta_idx ]
					else:
						for input_polarization_idx in range( 0, num_polarizations ):
							transmission_goal[ frequency_idx, input_polarization_idx, theta_idx, phi_idx, input_polarization_idx, order_idx ] = theta_transmission[ theta_idx ]



	return transmission_goal


def normal_amplification(
	ratio_normal_to_final, decay_theta, theta, weight_shape ):

	num_theta = len( theta )

	normal_weight = ratio_normal_to_final / ( 1 + ratio_normal_to_final )
	final_weight = 1 - normal_weight

	weight_by_theta = final_weight + ( normal_weight - final_weight ) * np.exp( -theta**2 / ( 2 * decay_theta ) )

	static_weights = np.zeros( weight_shape )

	for theta_idx in range( 0, num_theta ):
		static_weights[ :, :, theta_idx, :, :, : ] = weight_by_theta[ theta_idx ]

	return static_weights


def create_k_space_map( wavelength, img_mesh, img_dimension, numerical_aperture ):
	kr_max_um_inv = 2 * np.pi * ( img_dimension - 1 ) / ( img_dimension * img_mesh )
	k_values_um_inv = np.linspace( -0.5 * kr_max_um_inv, 0.5 * kr_max_um_inv, img_dimension )

	max_theta_rad = np.arcsin( numerical_aperture )

	theta_phi = []
	xy = []
	for x_idx in range( 0, img_dimension ):
		for y_idx in range( 0, img_dimension ):

			kx = k_values_um_inv[ x_idx ]
			ky = k_values_um_inv[ y_idx ]

			k = ( 2 * np.pi / wavelength )

			kz_sq = k**2 - kx**2 - ky**2

			if kz_sq < 0:
				continue

			kz = np.sqrt( kz_sq )
			kr = np.sqrt( kx**2 + ky**2 )

			theta_rad = np.arctan( kr / kz )
			phi_rad = np.arctan2( ky, kx )

			if theta_rad > max_theta_rad:
				continue

			theta_phi.append( [ theta_rad, phi_rad ] )
			xy.append( [ x_idx, y_idx ] )


	return theta_phi, xy


def create_random_phase_amplitude_samples( wavelength, img_mesh, img_dimension, angular_spread_size_radians, amplitude_spread_bounds, numerical_aperture, num_samples ):
	samples = np.zeros( ( num_samples, img_dimension, img_dimension ), dtype=npf.complex )
	# Output: ndarray "samples" of dimension (num_samples x img_dimension x img_dimension)

	max_theta_rad = np.arcsin( numerical_aperture )

	kr_max_um_inv = 2 * np.pi * ( img_dimension - 1 ) / ( img_dimension * img_mesh )
	k_values_um_inv = np.linspace( -0.5 * kr_max_um_inv, 0.5 * kr_max_um_inv, img_dimension )

	for sample_idx in range( 0, num_samples ):

		random_spatial_angle = angular_spread_size_radians * ( np.random.random( ( img_dimension, img_dimension ) ) - 0.5 )
		random_spatial_abs = amplitude_spread_bounds[ 0 ] + ( amplitude_spread_bounds[ 1 ] - amplitude_spread_bounds[ 0 ] ) * np.random.random( ( img_dimension, img_dimension ) )

		random_spatial = random_spatial_abs * np.exp( 1j * random_spatial_angle )

		k_space = np.fft.fftshift( np.fft.fft2( random_spatial ) )
		filter_k_space = np.zeros( k_space.shape, dtype=k_space.dtype )

		for x_idx in range( 0, img_dimension ):
			for y_idx in range( 0, img_dimension ):

				kx = k_values_um_inv[ x_idx ]
				ky = k_values_um_inv[ y_idx ]

				k = ( 2 * np.pi / wavelength )

				kz_sq = k**2 - kx**2 - ky**2

				if kz_sq < 0:
					continue

				kz = np.sqrt( kz_sq )
				kr = np.sqrt( kx**2 + ky**2 )

				theta_rad = np.arctan( kr / kz )
				if theta_rad > max_theta_rad:
					continue

				filter_k_space[ x_idx, y_idx ] = k_space[ x_idx, y_idx ]
		
		filter_random_spatial = np.fft.ifft2( np.fft.ifftshift( filter_k_space ) )

		samples[ sample_idx ] = filter_random_spatial


	return samples

# define_circular_zones_around_center(10, 100, 100, 15, 5)