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
	raise Exception('Please install NLOPT!')

import scipy
from scipy.ndimage import gaussian_filter
from scipy import optimize as scipy_optimize

# Neural Network
import torch
import torch.fft
import torchvision

# System, Plotting, etc.
import os
import pickle
import sys
import time
from datetime import datetime
import copy
from types import SimpleNamespace
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Parallel Computing
import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

comm = MPI.COMM_WORLD
resource_size = MPI.COMM_WORLD.Get_size()
processor_rank = MPI.COMM_WORLD.Get_rank()


# Custom Classes and Imports
from configs import *
from dataloader import *
from evaluation import *
from executor import *
from model import *
from utils import *

# Read in all the parameters
p = SimpleNamespace(**cfg_to_params.parameters)       #! Don't use 'p' for ANYTHING ELSE!
projects_directory_location = p.DATA_FOLDER

# Duplicate stdout to file
#TODO: Replace this with proper Python logging at some point
sys.stdout = utility.Logger(projects_directory_location)
program_starttime = time.time()

print("All modules loaded.")

#* Handle Input Arguments and Parameters
if len( sys.argv ) < 3:
	print( "Usage: srun python " + sys.argv[ 0 ] + " [ yaml file name ] [ \'eval\' or \'opt\' ] { override data folder }" )
	sys.exit( 1 )

# Determine run mode from bash script input arg
run_mode = sys.argv[ 2 ]
if not ( ( run_mode == 'eval' ) or ( run_mode == 'opt' ) ):
	print( 'Unrecognized mode!' )
	sys.exit( 1 )

# Override data folder given in YAML? If yes, third argument should be new data folder address
should_override_data_folder = False
override_data_folder = None
if len( sys.argv ) == 4:
	should_override_data_folder = True
	override_data_folder = sys.argv[ 3 ]

print("Input arguments processed.")


#* Read in all the parameters! (Hard-coded)
# This is probably the most Pythonic if tedious way to do it - but at least you can catch exceptions with dict.get()

parameter_hardcode_read_in = False
# 20230106 Ian - This is only in an if-statement block so I could collapse it in VSCode
if parameter_hardcode_read_in:
	random_seed = p.backend['random_seed']
	torch_backend_cudnn_enabled = p.backend['torch_backend_cudnn_enabled']
	
	#TODO: Data stuff
	img_dimension = p.data['img_dimension']
	img_mesh = p.data['img_mesh']
	
	number_dataset_samples = p.data['number_dataset_samples']
	number_training_samples = p.data['number_training_samples']
	number_validation_samples = p.data['number_validation_samples']
	training_batch_size = p.data['training_batch_size']
	batch_norm_full_network = p.data['batch_norm_full_network']
	
	train_batch_size = p.train['batch_size']
	train_buffer_size = p.train['buffer_size']
	train_epochs = p.train['epochs']
	train_val_subsplits = p.train['val_subsplits']
	optimizer = p.train['optimizer']
	metrics = p.train['metrics']
	
	test_batch_size = p.test['batch_size']
	test_log_interval = p.test['log_interval']
	
	#TODO: NN Model (even though there'll be no NN layers in the final version)
	
	# TODO: The rest of the stuff
	
	Q_ramp_start = p.optimization['Q_ramp_start']
	Q_ramp_end = p.optimization['Q_ramp_end']
	Q_ramp_style = p.optimization['Q_ramp_style']
	Q_number_steps = p.optimization['Q_number_steps']
	max_iterations_per_Q = p.optimization['max_iterations_per_Q']
	num_nn_epochs_per_photonic_epoch = p.optimization['num_nn_epochs_per_photonic_epoch']
	num_photonic_grad_descent_iter = p.optimization['num_photonic_grad_descent_iter']
	optimization_tolerance = p.optimization['optimization_tolerance']
	
	optimize_diffraction_efficiency = p.optimization['optimize_diffraction_efficiency']
	enable_method_of_moving_asymptotes = p.optimization['enable_method_of_moving_asymptotes']
	enable_method_of_moving_asymptotes_diffraction = p.optimization['enable_method_of_moving_asymptotes_diffraction']
	gradient_descent_step_size_normalized = p.optimization['gradient_descent_step_size_normalized']
	

#* Parameter Processing
# TODO: Move from here over to cfg_to_params.py

DEG_TO_RAD = np.pi / 180
p.simulation[ 'min_theta' ] = p.simulation[ 'min_theta_deg' ] * DEG_TO_RAD
p.simulation[ 'max_theta' ] = p.simulation[ 'max_theta_deg' ] * DEG_TO_RAD
p.simulation['angular_spread_size_rad'] = p.simulation['angular_spread_size_deg'] * DEG_TO_RAD

# c=1 in gRCWA; but this just boils down to f=1/λ[um]. Wavelengths submitted to gRCWA should be in um.
p.simulation['min_frequency'] = 1/p.simulation['max_wl']
p.simulation['max_frequency'] = 1/p.simulation['min_wl']

# Create (num_device_layers) design layers, all of the same thickness
p.device['design_layer_thicknesses'] = [p.device['design_layer_thickness'] for idx in range(0, p.device['num_device_layers'])]
# Create (num_device_layers - 1) spacer layers, all of the same thickness
p.device['spacer_layer_thicknesses'] = [p.device['spacer_layer_thickness'] for idx in range(0, p.device['num_device_layers'] - 1)]
p.device['spacer_layer_indices'] = [p.device['spacer_layer_index'] for idx in range(0, p.device['num_device_layers'] - 1)]

device_permittivity = p.device['device_index']**2
device_background_permittivity = p.device['device_background_index']**2

p.data['number_test_samples'] = p.data['number_dataset_samples'] -\
								p.data['number_training_samples'] - p.data['number_validation_samples']


#* Perform checks and assertions

if p.simulation['zero_order_weight'] is None:
	p.simulation['zero_order_weight'] = 1.0
weight_by_order = np.ones( p.simulation['num_orders'], dtype=np.float32 )
weight_by_order[ 0 ] = p.simulation['zero_order_weight']

if p.optimization['gradient_descent_step_size_normalized'] is None:
	p.optimization['gradient_descent_step_size_normalized'] = 0.01

if p.optimization['enable_method_of_moving_asymptotes'] is None:
	p.optimization['enable_method_of_moving_asymptotes'] = False
if p.optimization['enable_method_of_moving_asymptotes_diffraction'] is None:
	p.optimization['enable_method_of_moving_asymptotes_diffraction'] = False
if p.optimization['optimize_diffraction_efficiency'] is None:
	p.optimization['optimize_diffraction_efficiency'] = False

#TODO: Put all assertions on parameters here
# e.g. assert ( number_training_samples + number_validation_samples ) <= number_dataset_samples, "Too many samples in training and validation!"


frequencies = np.linspace( p.simulation['min_frequency'], p.simulation['max_frequency'], p.simulation['num_frequencies'] )
theta_values = np.linspace( p.simulation['min_theta'], p.simulation['max_theta'], p.simulation['num_theta'] )
numerical_aperture = np.sin( np.max( theta_values ) )

# Process frequency vector into k-values based on the image provided and the given NA
kr_max, kr_values = utility.kr_max( 1/frequencies[ 0 ], 
								   p.data['img_mesh'], p.data['img_dimension'], numerical_aperture )
# Form the k-grid
kx_values = kr_values.copy()
ky_values = kr_values.copy()
num_kx = len( kx_values )
num_ky = len( ky_values )

# Calculate padding for insertion and extraction of transmission map into RCWA
kx_pad = ( p.data['img_dimension'] - num_kx ) // 2
ky_pad = ( p.data['img_dimension'] - num_ky ) // 2

num_design_layers = len( p.device['design_layer_thicknesses'] )		# Just in case the array is processed to have more layers
Nx = p.simulation['num_planewaves'] - 1
Ny = Nx


total_dof = num_design_layers * Nx * Ny + p.device['number_additional_dof']
device_dof_end = num_design_layers * Nx * Ny


geometry = DeviceGeometry.DeviceGeometry(
			p.device['input_layer_thickness'], p.device['output_layer_thickness'], 
			p.device['input_index'], p.device['output_index'],
			[ Nx, Ny ], p.device['design_layer_thicknesses'], 
   			p.device['has_spacers'], p.device['spacer_layer_thicknesses'], p.device['spacer_layer_indices'],
			p.device['encapsulation_thickness'], p.device['encapsulation_index']
		)


#* Define transmission map by:
# frequency, input polarization, theta, phi, output polarization, order

# for now, dummy variable (initialise)
desired_transmission = np.zeros( ( 
							p.simulation['num_frequencies'], 
							p.simulation['num_polarizations'], 
							num_kx, 
							num_ky, 
							p.simulation['num_polarizations_out'], 
							p.simulation['num_orders']
						), dtype=complex )
# TODO: Remove the relationship to the number of freqs., polarizations, etc.

# Initialize dynamic and static weights
dynamic_weights = ( 1. / np.product( desired_transmission.shape ) ) * np.ones( desired_transmission.shape )

static_weights = None
# if static_weights_filepath is not None:
# 	static_weights = np.load( static_weights_filepath )
static_weights = np.ones(desired_transmission.shape)
# TODO: Add more nuanced static weight definitions later on


#* Job Allocation for Running in Parallel
#! Surely there must be a way to not have to run this part of the code in every process, just the master one?
# TODO: See about offloading this to some function or other

# For each Q, we are running n_freq * n_pols * n_kx * n_ky jobs. We will iterate through the entire value space for each of these parameters.
num_jobs = p.simulation['num_frequencies'] * p.simulation['num_polarizations'] * num_kx * num_ky
# How many resources do we have? That is determined by resource_size = MPI.COMM_WORLD.Get_size()
min_jobs_per_resource = int( np.floor( num_jobs / resource_size ) )
# Create an array defining the # of jobs for each resource, and how many jobs are left over.
jobs_per_resource = [ min_jobs_per_resource for idx in range( 0, resource_size ) ]
remaining_jobs = num_jobs - min_jobs_per_resource * resource_size
# Distribute the remaining jobs among the available resources. For e.g. 6 jobs, 4 resources -> [2, 2, 1, 1]
for idx in range( 0, remaining_jobs ):
	jobs_per_resource[ idx ] += 1

# Create index arrays for all the degrees of freedom we are iterating over for each Q
frequency_opt_idxs = np.arange( 0, p.simulation['num_frequencies'] )
polarizations_opt_idxs = np.arange( 0, p.simulation['num_polarizations'] )
kx_opt_idxs = np.arange( 0, num_kx )
ky_opt_idxs = np.arange( 0, num_ky )
all_opt_idxs = np.meshgrid( frequency_opt_idxs, polarizations_opt_idxs, kx_opt_idxs, ky_opt_idxs )

# Flatten out the ENTIRE index array into a 1D vector for access in the below block:
flatten_opt_idxs = []
for frequency_idx in range( 0, p.simulation['num_frequencies'] ):
	for polarization_idx in range( 0, p.simulation['num_polarizations'] ):
		for kx_idx in range( 0, num_kx ):
			for ky_idx in range( 0, num_ky ):

				flatten_opt_idxs.append( ( frequency_idx, polarization_idx, kx_idx, ky_idx ) )

# Note down which job index each resource will begin at. Going off the above example, resource_start_idxs will be [0, 2, 4, 5, 6]
resource_start_idxs = []
resource_start_idx = 0
for job_idx in range( 0, resource_size ):
	resource_start_idxs.append( resource_start_idx )
	resource_start_idx += jobs_per_resource[ job_idx ]

# The rank of the current process running this code is assigned by MPI. 
# Now it is going to know which job idxs it has been assigned to run.
jobs_for_resource_start = resource_start_idxs[ processor_rank ]
# When an optimization loop happens, these are the jobs to run on a given processor.
jobs_for_resource = flatten_opt_idxs[ jobs_for_resource_start : ( jobs_for_resource_start + jobs_per_resource[ processor_rank ] ) ]
# For e.g. if processor_rank = 1, jobs_for_resource will be [2,3]


#* Functions to Enable Optimization:
def run_single_simulation(
	frequency_idx_, polarization_idx_, kx_idx_, ky_idx_, Qabs_, device_permittivity_,
	weights_, transmission_map_device_, figure_of_merit_individual_,
	gradient_transmission_polarization_and_order_ ):#,
	# desired_transmission_phase_by_frequency_and_order_ ):
	'''Runs a single simulation through RCWA, with the relevant parameters passed as input arguments.
 	Outputs: figure of merit for this particular simulation'''
	
	# 20230104, Ian - At present this is only ever called via the following:
	# figure_of_merit_total += run_single_simulation(
	# 				frequency_idx, polarization_idx, kx_idx, ky_idx,
	# 				optimization_Qabs, device_permittivity_,
	# 				optimization_weights,
	# 				transmission_map_device, figure_of_merit_individual,
	# 				gradient_transmission_polarization_and_order )

	# Access the specific values of parameters for this simulation
	frequency = frequencies[ frequency_idx_ ]
	# frequency_Qabs attenuates higher-frequency resonances and introduces some loss
	frequency_Qabs = frequency * ( 1 + 1j / ( 2 * Qabs_ ) )		# See ex1.py, https://github.com/weiliangjinca/grcwa/tree/master/example
 
	polarization = p.simulation['polarizations'][ polarization_idx_ ]
	# theta = theta_values[ theta_idx_ ]
	# phi = phi_values[ phi_idx_ ]
	kx = kx_values[ kx_idx_ ]
	ky = ky_values[ ky_idx_ ]
	
	#* Simulation Setup
	# Calculate kz, kr and then obtain theta, phi of incident planewave
	k = ( 2 * np.pi * frequencies[ frequency_idx_ ] )	# Remember in gRCWA, c=1
	kz_sq = k**2 - kx**2 - ky**2
	kz = np.sqrt( kz_sq )
	kr = np.sqrt( kx**2 + ky**2 )
	theta = np.arctan( kr / kz )
	phi = np.arctan2( ky, kx )
 
	# Define planewave via dictionary (see ex2.py, https://github.com/weiliangjinca/grcwa/tree/master/example)
	planewave = { 'p_amp':0, 's_amp':1, 'p_phase':0, 's_phase':0 }
	if polarization == 'p':
		planewave = { 'p_amp':1, 's_amp':0, 'p_phase':0, 's_phase':0 }

 
	# Create gRCWA environment
	simulation = grcwa.obj( p.simulation['num_planewaves'], p.simulation['lattice_x'], p.simulation['lattice_y'],
							frequency_Qabs, theta, phi, verbose=0 )
	# Add layers according to DeviceGeometry object
	geometry.add_layers( simulation )
	# Perform initial setup (mandatory for gRCWA)
	simulation.Init_Setup()
	# Feed the epsilon profile for patterned layer, https://grcwa.readthedocs.io/en/latest/usage.html
	# device_permittivity_ needs to be a flattened 1D array.
	simulation.GridLayer_geteps( device_permittivity_ )
	# Create planewave
	simulation.MakeExcitationPlanewave(
		planewave['p_amp'],
		planewave['p_phase'],
		planewave['s_amp'],
		planewave['s_phase'],
		order = 0 )
	
 
	#* Get the phase of the input and outputs
 	# Obtain amplitudes of Fourier eigenvectors, at some layer at some zoffset:
	input_fourier = simulation.GetAmplitudes( 0, geometry.input_fourier_sampling_offset ) [ 0 ]
	forward_fourier = simulation.GetAmplitudes( simulation.Layer_N - 1, geometry.output_fourier_sampling_offset ) [ 0 ]
	# At phi = 0 degrees, the input_fourier corresponds to [s-data, p-data] (CONCATENATED!)
 	# At phi = 90 degrees, the input_fourier corresponds to [p-data, s-data]
 
	output_length = len( forward_fourier ) // 2
	get_s_lattice_input_data = input_fourier[ 0 : output_length ]
	get_p_lattice_input_data = input_fourier[ output_length : ]
	get_s_lattice_output_data = forward_fourier[ 0 : output_length ]
	get_p_lattice_output_data = forward_fourier[ output_length : ]
 
	# Convert to the appropriate transmission information.
	#! This is due to a quirk of S4, which gRCWA largely copies in its simulation methods.
	# It's one of the references in here that explains it: https://pubs.acs.org/doi/10.1021/acsphotonics.0c00768
	convert_input_s_data = np.cos( phi ) * get_s_lattice_input_data + np.sin( phi ) * get_p_lattice_input_data
	convert_input_p_data = -np.sin( phi ) * get_s_lattice_input_data + np.cos( phi ) * get_p_lattice_input_data

	convert_output_s_data = np.cos( phi ) * get_s_lattice_output_data + np.sin( phi ) * get_p_lattice_output_data
	convert_output_p_data = -np.sin( phi ) * get_s_lattice_output_data + np.cos( phi ) * get_p_lattice_output_data

	# Note: the below is only normalizing based on input polarization and assuming the zeroth order transmission.
	# To account for the full normalization, we need to adjust based on the output theta.  For now, we won't consider this
	# added complexity.
 
	if polarization == 's':
		normalize_output_s = convert_output_s_data / ( convert_input_s_data[ 0 ] + np.finfo( np.float32 ).eps )
		normalize_output_p = convert_output_p_data / ( convert_input_s_data[ 0 ] + np.finfo( np.float32 ).eps )
		#! See above quirk
		# normalize_output_s *= ( geometry.input_index * np.cos( theta_incident ) / ( geometry.output_index * np.cos( theta_transmitted ) ) )
		# normalize_output_p *= ( geometry.input_index * np.cos( theta_incident ) / ( geometry.output_index * np.cos( theta_transmitted ) ) )
	else:
		normalize_output_s = convert_output_s_data / ( convert_input_p_data[ 0 ] + np.finfo( np.float32 ).eps )
		normalize_output_p = convert_output_p_data / ( convert_input_p_data[ 0 ] + np.finfo( np.float32 ).eps )
		#! See above quirk
		# normalize_output_s *= ( geometry.input_index / geometry.output_index )
		# normalize_output_p *= ( geometry.input_index / geometry.output_index )
	
	if p.simulation['normalize_to_Fresnel']:
		theta_incident = theta
		# Snell it
		theta_transmitted = np.arcsin( ( geometry.input_index / geometry.output_index ) * np.sin( theta_incident ) )
		normalize_output_s *= ( geometry.input_index * np.cos( theta_incident ) / ( geometry.output_index * np.cos( theta_transmitted ) ) )
		normalize_output_p *= ( geometry.input_index / geometry.output_index )
	
	normalize_output_by_polarization = [ normalize_output_s, normalize_output_p ]
	
	#
	# This should be an array that is of size 2 x N_orders where N_orders is the number of orders being optimized over.
	# We assume the s-polarization output goal is first.
	# 
	optimization_goal_slice = gradient_transmission_polarization_and_order_[ frequency_idx_, polarization_idx_, kx_idx_, ky_idx_ ]
	optimization_weights_slice = weights_[ frequency_idx_, polarization_idx_, kx_idx_, ky_idx_ ]
 
	figure_of_merit = 0.0
	for polarization_out_idx in range( 0, len( normalize_output_by_polarization ) ):
		for order_idx in range( 0, p.simulation['num_orders'] ):
			
			# Now we have optimization_goal_slice = δ(FoM)/δε. We want δ(FoM) = δε * optimization_goal_slice
			# Explanation: For two complex numbers, Re(z1*z2) = Re(z1)*Re(z2) + [-Im(z1)]*Im(z2)
			real_opt_slice = 0.5 * ( optimization_goal_slice[ polarization_out_idx, order_idx ] + np.conj( optimization_goal_slice[ polarization_out_idx, order_idx ] ) )
			imag_opt_slice = ( 0.5 / 1j ) * ( optimization_goal_slice[ polarization_out_idx, order_idx ] - np.conj( optimization_goal_slice[ polarization_out_idx, order_idx ] ) )
			# individual_fom = np.real( real_opt_slice * np.real( fake_t ) + imag_opt_slice * np.imag( fake_t ) )
			individual_fom = np.real( real_opt_slice * np.real( normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] ) + imag_opt_slice * np.imag( normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] ) )
			
   			# Weight accordingly
			individual_fom_weighted = (
				optimization_weights_slice[ polarization_out_idx, order_idx ] * individual_fom )

			# Store fom in the array
			# If statements account for some value types 
			if isinstance( individual_fom, float ):
				figure_of_merit_individual_[ frequency_idx_, polarization_idx_, kx_idx_, ky_idx_, polarization_out_idx, order_idx ] = individual_fom
			else:
				figure_of_merit_individual_[ frequency_idx_, polarization_idx_, kx_idx_, ky_idx_, polarization_out_idx, order_idx ] = individual_fom._value

			# Store transmission in the map array
			pull_t = normalize_output_by_polarization[ polarization_out_idx ][ order_idx ]
			if isinstance( pull_t, complex ):
				transmission_map_device_[ frequency_idx_, polarization_idx_, kx_idx_, ky_idx_, polarization_out_idx, order_idx ] = pull_t
			else:
				transmission_map_device_[ frequency_idx_, polarization_idx_, kx_idx_, ky_idx_, polarization_out_idx, order_idx ] = pull_t._value


			# Add the weighted fom here so that the autograd picks it up and weights the gradient accordingly.
			figure_of_merit += individual_fom# individual_fom_weighted

		
	return figure_of_merit

def processor_simulate(data):
	'''This function is called when a worker or master process receives a message with 'simulate' instructions.
	Along with the instruction, the process will have receide some other information:
	combined weighting function for all figures of merit, and the current optimization variable.
	Input: data - dictionary containing all of the above.
	Output: results - dictionary containing relevant information extracted from each EM simulation.'''
	
	optimization_weights = data[ 'weights' ]
	optimization_variable = data[ 'optimization_variable' ]
	optimization_Qabs = data[ 'Qabs' ]
	gradient_transmission_polarization_and_order = data[ 'gradient_transmission_polarization_and_order' ]

	transmission_map_device = np.zeros( desired_transmission.shape, dtype=complex )
	figure_of_merit_individual = np.zeros( desired_transmission.shape )

	# Define a function to get the figure of merit for each of the jobs assigned, and then total it.
	def run_jobs_for_resource(optimization_variable_):
		'''
		Looks at the total jobs for each resource - each job will have different param. values for frequency, polarization, kx, ky etc.
		Runs a single simulation for each of those values, and adds to the figure of merit.
		Input: Permittivity (optimization variable) at this point. Every job gets the same optimization variable values.
  		Output: figure_of_merit_total'''
		
		# Initialize total FoM
		figure_of_merit_total = 0.0
		# Apply permittivity symmetry constraints to permittivity block (optimization variable)
		device_permittivity_ = preprocess_optimization_variable( optimization_variable_ )
	
		# The job idxs for this particular process (resource) are defined above in the parallelization segment. 
		for job_idx in range( 0, len( jobs_for_resource ) ):
			frequency_idx, polarization_idx, kx_idx, ky_idx = jobs_for_resource[ job_idx ]

			figure_of_merit_total += run_single_simulation(
				frequency_idx, polarization_idx, kx_idx, ky_idx,
				optimization_Qabs, device_permittivity_,
				optimization_weights,
				transmission_map_device, figure_of_merit_individual,
				gradient_transmission_polarization_and_order )
		
  
		return figure_of_merit_total
	
 	# Autograd will calculate the gradient of that total figure of merit.
	run_jobs_for_resource_grad = grad( run_jobs_for_resource )
	# VSCode will say code after this is unreachable, but that is not true
 
	fom_for_resource = run_jobs_for_resource( optimization_variable )
	net_grad_for_resource = run_jobs_for_resource_grad( optimization_variable )

	results = {
		"instruction" : "results",
		"processor_rank" : processor_rank,
		"figure_of_merit" : fom_for_resource,
		"net_grad" : net_grad_for_resource,
		"figure_of_merit_individual" : figure_of_merit_individual,
		"transmission_map_device" : transmission_map_device
	}


	# Nonblocking send for the case where you are sending from rank 0 to rank 0
	if not ( processor_rank == 0 ):
		comm.isend( results, dest=0 )
	
	return results

def evaluate_k_space( Qabs_, optimization_variable_, gradient_transmission_polarization_and_order ):
	'''Called from the main optimization loop for each value of Qabs_ (the iteration variable).
	[What does this do?]
	Inputs: Qabs (iteration variable), 
	optimization_variable_ (permittivity of device at that iteration),
	gradient_transmission_polarization_and_order (gradient of FoM with respect to permittivity at that iteration).
	Outputs: figure_of_merit, gradn, figure_of_merit_individual, transmission_map_device'''
	
	global dynamic_weights
	global desired_transmission
	
	# Initialize outputs
	gradn = np.zeros( optimization_variable_.shape, dtype=optimization_variable_.dtype )
	figure_of_merit = 0.0
	figure_of_merit_individual = np.zeros( desired_transmission.shape )
	transmission_map_device = np.zeros( desired_transmission.shape, dtype=complex )
	
	# Create dictionary to pass to worker processes through MPI
	optimize_message = {
		"instruction" : "simulate",
		"weights" : dynamic_weights * static_weights,
		"optimization_variable" : np.array( optimization_variable_, dtype=np.float64 ),
		# Need to send out some gradient information
		"gradient_transmission_polarization_and_order" : np.array( gradient_transmission_polarization_and_order, dtype=complex ),
		"Qabs" : Qabs
	}
	
 	# Blocking send to all worker processes. (No sends to the rank 0 processor, which would cause a lock-up)
	# Sending this message initiates a simulation through processor_simulate()
	for processor_rank_idx_ in range( 1, resource_size ):
		comm.send( optimize_message, dest=processor_rank_idx_ )
	data = optimize_message
 
	# We now loop through each process and build up the values for FoMs and gradients.
	for processor_rank_idx_ in range( 0, resource_size ):
	 
		# Get back the result of the simulation for each process. Each results variable is a dictionary with keys
		if processor_rank_idx_ == 0:
			results = processor_simulate( data )
		else:
			results = comm.recv( source=processor_rank_idx_ )

		# Account for an erroneous or failed simulation!
		if not ( results[ "instruction" ] == "results" ):
			print( "Unexpected message received from processor " + str( processor_rank_idx_ ) + "\nExiting..." )
			sys.exit( 1 )


		gradn[:] += results[ "net_grad" ]		# Indexing is used to overwrite grad.
		# See https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#:~:text=as%20described%20below.-,Assigning%20results%20in%2Dplace,-Your%20objective%20and
		figure_of_merit += results[ "figure_of_merit" ]
		figure_of_merit_individual_slice = results[ "figure_of_merit_individual" ]
		transmission_map_device_slice = results[ "transmission_map_device" ]
		
		# This means the jobs for the resource that is currently being accessed with rank processor_rank_idx.
		jobs_for_other_resource = flatten_opt_idxs[
				resource_start_idxs[ processor_rank_idx_ ] : 
				(resource_start_idxs[ processor_rank_idx_ ] + jobs_per_resource[ processor_rank_idx_ ])
			]
		
		# For each job...
		for job_idx in range( 0, len( jobs_for_other_resource ) ):
			frequency_idx, polarization_idx, kx_idx, ky_idx = jobs_for_other_resource[ job_idx ]

			# For each (output) polarization and order...
			for polarization_out_idx in range( 0, p.simulation['num_polarizations_out'] ):
				for order_idx in range( 0, p.simulation['num_orders'] ):
		
					# We add to the figure of merit individual array.
					figure_of_merit_individual[
						frequency_idx, polarization_idx, kx_idx, ky_idx, polarization_out_idx, order_idx ] = figure_of_merit_individual_slice[
							frequency_idx, polarization_idx, kx_idx, ky_idx, polarization_out_idx, order_idx ]
	  
					# We add to the transmission map individual array.
					transmission_map_device[
						frequency_idx, polarization_idx, kx_idx, ky_idx, polarization_out_idx, order_idx ] = transmission_map_device_slice[
							frequency_idx, polarization_idx, kx_idx, ky_idx, polarization_out_idx, order_idx ]

	
	return figure_of_merit, gradn, figure_of_merit_individual, transmission_map_device

def preprocess_permittivity( device_permittivity ):
	'''Apply predefined processing filters (e.g. symmetry constraints) to the device permittivity block.'''
	
	#TODO:  At some point this should turn into the preprocessing filter chain (i.e. - take a density variable to a permittivity one)
	if p.simulation['permittivity_symmetry_function'] is not None:
		if p.simulation['permittivity_symmetry_function'] == "c4_symmetry_explicit":
			device_permittivity = utility.c4_symmetry_explicit( device_permittivity, Nx, Ny, num_design_layers )
		elif p.simulation['permittivity_symmetry_function'] == "vertical_flip_symmetry_explicit":
			device_permittivity = utility.vertical_flip_symmetry_explicit( device_permittivity, Nx, Ny, num_design_layers )
		elif p.simulation['permittivity_symmetry_function'] == "woodpile":
			device_permittivity = utility.woodpile( device_permittivity, Nx, Ny, num_design_layers )
		elif p.simulation['permittivity_symmetry_function'] == "woodpile_and_vertical_flip_symmetry_explicit":
			device_permittivity = utility.vertical_flip_symmetry_explicit(
				utility.woodpile( device_permittivity, Nx, Ny, num_design_layers ), Nx, Ny, num_design_layers )
		else:
			print( 'Unrecognized permittivity symmetry function' )
			sys.exit( 1 )

	return device_permittivity

def process_additional_dof( additional_dof_ ):
	desired_transmission_phase_by_frequency_and_order = np.zeros( ( p.simulation['num_frequencies'],
																	p.simulation['num_orders'] ), 
															  dtype=complex )

	if p.simulation['fom_adjustment_function'] is not None:
		if p.simulation['fom_adjustment_function'] == "modify_transmission_flat_phase":
			desired_transmission_phase_by_frequency_and_order = np.reshape( additional_dof_, ( 
																	p.simulation['num_frequencies'],
																	p.simulation['num_orders'] 
																 	) 
																)
		else:
			print( 'Unrecognized FoM adjustment function' )
			sys.exit( 1 )

	return np.exp( 1j * desired_transmission_phase_by_frequency_and_order )

def preprocess_optimization_variable( optimization_variable_ ):
	device_permittivity = preprocess_permittivity( optimization_variable_ )
	return device_permittivity

# =========================================================================================================================

#* Main Optimization Code Loop
# Things start breaking off by master job and worker jobs for the parallel
# compute.  The master is responsible for handling the optimization and the
# workers are responsible for running simulations given the current optimization
# state and then sending back gradient and figure of merit information to the master.
# For more information: https://rabernat.github.io/research_computing/parallel-programming-with-mpi-for-python.html



#* Let the master job have rank 0-----------------------------------------------------------------------------------------------

if processor_rank == 0:
	
	# Configure modules based on parameters
	torch.manual_seed(p.backend['random_seed'])
	torch.backends.cudnn.enabled = p.backend['torch_backend_cudnn_enabled']        # Disables nondeterministic algorithms for cuDNN
	# A basic neural net tutorial using PyTorch is provided here as handy reference:
 	# https://pyimagesearch.com/2021/07/12/intro-to-pytorch-training-your-first-neural-network-using-pytorch/
	
 
	#* Create the datasets and wrap them with DataLoaders
	# Guide to DataLoaders: https://blog.paperspace.com/dataloaders-abstractions-pytorch/
	# Custom Datasets and DataLoaders: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

	# Enable GPU handling if possible
	worker_kwargs = {'num_workers': 0, 'pin_memory': False}
	if torch.cuda.is_available():
		hardware_config = "cuda"
		worker_kwargs = {'num_workers': 1, 'pin_memory': True} 
	else:	hardware_config = "cpu"

	dataset_kwargs = {'name': p.data['name'], 'img_dimension': p.data['img_dimension']}
	
	dataset_samples = datasets.get_processed_dataset(True, **dataset_kwargs)
	training_samples = torch.utils.data.Subset(dataset_samples, range(0, p.data['number_training_samples']))
	validation_samples = torch.utils.data.Subset(dataset_samples, range(p.data['number_training_samples'],
									p.data['number_training_samples'] + p.data['number_validation_samples']))
 
	training_samples_targets = dataset_samples.targets[ range(0, p.data['number_training_samples']) ]
	validation_samples_targets = dataset_samples.targets[ range(p.data['number_training_samples'],
								p.data['number_training_samples'] + p.data['number_validation_samples']) ]
 
	train_loader = torch.utils.data.DataLoader(training_samples, batch_size=p.train['batch_size'], shuffle=True, 
											**worker_kwargs)
	validation_loader = torch.utils.data.DataLoader(validation_samples, batch_size=p.train['batch_size'], shuffle=True, 
											**worker_kwargs)
 
	test_samples = datasets.get_processed_dataset(False, **dataset_kwargs)
	test_loader = torch.utils.data.DataLoader(test_samples, batch_size=p.validation['batch_size'], shuffle=True, 
												 **worker_kwargs)
 
	# How many training batches are there?
	p.data['number_training_samples'] = len(training_samples)		# Overwrites whatever is in the config if the dataset is standard
	p.data['number_validation_samples'] = len(validation_samples)	# And if it's non-standard, then this variable doesn't change
	number_batches = ( p.data['number_training_samples'] // p.data['training_batch_size'] ) + \
					( ( p.data['number_training_samples'] % p.data['training_batch_size'] ) > 0 )
	
	# Based on that, identify the idxs at which each training batch is going to start and end.
	training_batch_start_idxs = [ p.data['training_batch_size'] * batch_idx for batch_idx in range( 0, number_batches ) ]
	training_batch_end_idxs = [ np.minimum(
									training_batch_start_idxs[ batch_idx ] + p.data['training_batch_size'],
									p.data['number_training_samples'] 
									)
								for batch_idx in range( 0, number_batches ) ]
	
 
	#* Fourier transforming the dataset images is not a trivial task (code-wise).
	# 20230109 Ian - This is what I ended up doing: https://stackoverflow.com/a/59661024
	# mixed with https://laurentperrinet.github.io/sciblog/posts/2018-09-07-extending-datasets-in-pytorch.html
	#! Can also consider using a Lambda transform from torch: https://stackoverflow.com/questions/60900406/torchvision-transforms-implementation-of-flatten
	# Either of these methods can be used to implement additional perturbations and transformations (shift, intensity change, rotation etc.)
 
	#! TODO: There shouldn't be two different behaviours for diff. run_modes.
 	#! I've only turned it off so that debugging won't take forever.
	#! Also enable test_Ek again at some point, please.
	training_Ek = datasets.fourier_transform_dataset_with_NA(training_samples, training_samples_targets, dataset_samples.transform,
															kx_values, ky_values, frequencies, numerical_aperture)
	validation_Ek = datasets.fourier_transform_dataset_with_NA(validation_samples, validation_samples_targets, dataset_samples.transform,
															kx_values, ky_values, frequencies, numerical_aperture)
	# test_Ek = datasets.fourier_transform_dataset_with_NA(test_samples, test_samples.targets, test_samples.transform,
	# 														kx_values, ky_values, frequencies, numerical_aperture)
	# if run_mode == 'eval':
	# 	training_Ek = datasets.CustomDataset(training_samples.dataset.data, training_samples.dataset.targets, training_samples.dataset.transform)
	# 	validation_Ek = datasets.CustomDataset(validation_samples.dataset.data, validation_samples.dataset.targets, validation_samples.dataset.transform)
 
 
	# Split dataset into training, validation and test samples
	ground_truth_training = copy.deepcopy(training_samples_targets)
	ground_truth_validation = copy.deepcopy(validation_samples_targets)

	# Generate the zones at the detector plane that will correspond to each of the classes.
	target_intensity_distributions = utility.target_classes_to_zones(dataset_samples.classes, p.data['img_dimension'])

	#* Optimization variable initialization and bounds
	init_optimization_variable = np.zeros( total_dof )

	lower_bounds = np.ones( total_dof, dtype=float )
	upper_bounds = np.ones( total_dof, dtype=float )
	lower_bounds[ 0 : device_dof_end ] = device_background_permittivity
	upper_bounds[ 0 : device_dof_end ] = device_permittivity
	
	# Device Permittivity Initialization
	
	init_permittivity = np.zeros( Nx * Ny * num_design_layers )
	for layer_idx in range( 0, num_design_layers ):
		blank_density = np.zeros( ( Nx, Ny ) )

		if p.device['device_initialization'] in ['random']:
			init_layer_density = np.random.random( blank_density.shape )
			init_layer_density = gaussian_filter( init_layer_density, sigma=p.device['initialization_blur_sigma'] )

			init_layer_density = init_layer_density - np.mean( init_layer_density )# + initialization_mean
			init_layer_density = init_layer_density * p.device['initialization_std'] / np.std( init_layer_density )
			init_layer_density += p.device['initialization_mean']
			init_layer_density = np.minimum( 1.0, np.maximum( init_layer_density, 0.0 ) )

		elif p.device['device_initialization'] in ['uniform']:
			init_layer_density = p.device['initialization_uniform_density'] * np.ones( blank_density.shape )
		else:
			print( 'Unrecognized device initialization strategy!' )
			sys.exit( 1 )
		
  		# Rescale the initial layer density to the background permittivity and allowed maximum permittivity
		init_layer_permittivity = device_background_permittivity + \
  								( device_permittivity - device_background_permittivity ) * init_layer_density
		# Put it into the corresponding layer
		init_permittivity[ layer_idx * Nx * Ny : ( layer_idx + 1 ) * Nx * Ny ] = init_layer_permittivity.flatten()

	init_optimization_variable[ 0 : device_dof_end ] = init_permittivity
	#NOTE Q: Why is this a 1D variable? Why are we flattening?
 	# Ans: when passing permittivity to gRCWA, instead of 3D permittivity voxel block,
  	# it requires a 1D flattened array in the above fashion.
 

	# Extra dof initializations - right now we assume all extra degrees of freedom are phases for transmission coefficients.
	if p.device['number_additional_dof'] > 0:
		
		if p.device['init_function_additional_dof'] in ['init_phase_dof']:
			init_optimization_variable[ device_dof_end : ] = np.pi
		else:
			print( 'Unrecognized additional dof strategy!' )
			sys.exit( 1 )

		if p.device['bounds_function_additional_dof'] in ['bounds_phase_dof']:
			lower_bounds[ device_dof_end : ] = 0
			upper_bounds[ device_dof_end : ] = 2 * np.pi
		else:
			print( 'Unrecognized additional dof strategy!' )
			sys.exit( 1 )
   
   
	#* Here we branch off depending on whether we are optimizing or evaluating.----------------------------------
 
	if run_mode == 'opt':
		
		iteration_counter = 0
		iteration_counter_diffraction = 0
		# optimization_variable = init_optimization_variable.copy()
		
		# transmission_map_device_ = torch.ones( desired_transmission.shape, dtype=torch.complex64, requires_grad=True )
		transmission_map_device_ = np.ones( desired_transmission.shape, dtype=complex )
		optimization_variable = transmission_map_device_.copy()

		# Qabs is a parameter for relaxation to better approach global optimal, at Qabs = inf, it will describe the real physics.
		# It can also be used to resolve any singular matrix issues by setting a large but finite Qabs, e.g. Qabs = 1e5.
		Q_values_optimization = None
		if p.optimization['Q_ramp_style'] == 'linear':
			Q_values_optimization = utility.Q_ramp_linear(  p.simulation['Q_ramp_start'],
												 			p.simulation['Q_ramp_end'],
															p.simulation['Q_number_steps'] )
		elif p.optimization['Q_ramp_style'] == 'exponential':
			Q_values_optimization = utility.Q_ramp_exponential( p.optimization['Q_ramp_start'],
																p.optimization['Q_ramp_end'],
																p.optimization['Q_number_steps'] )
		else:
			print( 'Unknown Q ramp style!' )
			sys.exit( 1 )
		# We will always end on Q=inf to simulate real life e.g. aperiodic context
		Q_values_optimization = np.append(Q_values_optimization, np.inf)
  

		# TODO: construct a net from the architecture in the config yaml, not hardcode like below

		# Define NN model
		nn_model = networks.identity_net(p.simulation['num_orders'])		 # Literally no NN architecture to optimize at all
		# nn_model = networks.simple_net(p.simulation['num_orders'])		# TODO: Turn off!!
		
		# Define NN Optimizer
		if p.train['optimizer']['type'].upper() in ['SGD']:
			nn_optimizer = torch.optim.SGD( nn_model.parameters(), 
									lr = p.train['optimizer']['learning_rate'], 
									momentum = p.train['optimizer']['momentum']
									)
		elif p.train['optimizer']['type'].lower() in ['adam']:
			nn_optimizer = torch.optim.Adam( nn_model.parameters(), 
									lr = p.train['optimizer']['learning_rate']
								 )
		else:
			print( 'Unknown optimizer type!' )
			sys.exit( 1 )

		# Define loss function
		nn_loss_fn = torch.nn.MSELoss()
		# nn_loss_fn = torch.nn.L1Loss()
  
  
		#* Determine Q iteration values (optimization relaxation variable)) -----------------------------------------------------
		# Qabs is a parameter for relaxation to better approach global optimal, at Qabs = inf, it will describe the real physics.
		# It can also be used to resolve any singular matrix issues by setting a large but finite Qabs, e.g. Qabs = 1e5.
		num_Q_iterations = len(Q_values_optimization)
		training_error = np.zeros( num_Q_iterations )
		validation_error = np.zeros( num_Q_iterations )
		test_error = np.zeros( num_Q_iterations )
		test_accuracy = np.zeros( num_Q_iterations )

		for Q_idx in range( 0, num_Q_iterations ):
			# For each value of Q, we create an initial uniform gradient map (with the same shape as the transmission map).
			# We are using ones as the seed gradient for the optimization.
			Qabs = Q_values_optimization[ Q_idx ]
			print(f'Moving to Q_abs={Qabs:.3f}: Value {Q_idx}/{num_Q_iterations}')
			print("Current Time: " + datetime.now().strftime(r"%d/%m/%Y %H:%M:%S"))
			print(f'Program Timer: {(time.time() - program_starttime):.1f}' + "\n")
   
			blank_gradient = np.ones( desired_transmission.shape, dtype=complex )
   
			# # We input the current Q-value, the current permittivity optimization variable, and the blank gradient map.
			# # This function runs simulations and gets the FoM for every single permutation of the degrees of freedom being surveyed:
			# # freqs., polarizations, orders, etc. etc. etc.
			# # It returns also the gradient calculated through autograd.
			# figure_of_merit, gradn, figure_of_merit_individual_, transmission_map_device_ =\
		  	# 		evaluate_k_space( Qabs, optimization_variable, blank_gradient )
	 
			# Normalization is going to change by minibatch and current device - put a batch norm on the input intensities
			# https://arxiv.org/abs/1805.11604
			# https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
			photonic_batch_idx = 0
			photonic_batch_idx_diffraction = 0
			# These idxs are tied to the line <for idx in range( 0, p.optimization['num_photonic_grad_descent_iter'] )>:
			# i.e. when MMA is NOT selected and the custom function function_opt_X is called.

			#! 20230109 Ian - This is where we diverge from the [v3] phase reconstruction code!!!
   
			gradn = np.zeros( optimization_variable.shape, dtype=optimization_variable.dtype )
			figure_of_merit = 0.0
			figure_of_merit_individual = np.zeros( desired_transmission.shape )
			#! 20230109 Ian - Take transmission_map_device out of the loop since it's what we're optimizing directly.
			# transmission_map_device_ = torch.ones( desired_transmission.shape, dtype=torch.complex64, requires_grad=True )
   
			#* Training Loop for ONLY the NN model side for this value of Q_abs
			# Calling the train() method of the PyTorch model is required for the model parameters 
   			# to be updated during backpropagation.
			nn_model.train()
			# Run through a couple of rounds of training here - this will get the batch norm going!
			nn_model.train()
   
			log_file = open( p.DATA_FOLDER + "/log.txt", 'a' )

			# We are going to train the NN model side for this value of Q_abs.
			# NOTE: This is a fresh, new NN model that bears no connection to previous ones
			# except for the optimization_variable (permittivity) that has now been updated.
			for nn_epoch in range( 0, p.optimization['num_nn_epochs_per_photonic_epoch'] ):
				average_loss = 0
	
				for nn_batch_idx in range( 0, number_batches ):
					if nn_batch_idx % 20 == 0:
						print(f'Neural Network Epoch {nn_epoch}, Batch {nn_batch_idx}/{number_batches}')
		
					batch_start_idx = training_batch_start_idxs[ nn_batch_idx ]
					batch_end_idx = training_batch_end_idxs[ nn_batch_idx ]
					number_in_batch = batch_end_idx - batch_start_idx

					# Now we pull up data for the NN: the output_I, calculated from
	 				# - the input_tmap (we got it from running evaluate_k_space() again)
					# - the input_Ek
					# input_tmap = np.pad(
					# 		transmission_map_device_,
					# 		(
					# 			( 0, 0 ), ( 0, 0 ),
					# 			( kx_pad, kx_pad ),
					# 			( ky_pad, ky_pad ),
					# 			( 0, 0 ), ( 0, 0 ) ),
					# 		mode='constant' )
					input_tmap = torch.nn.functional.pad(
							transmission_map_device_,
							(0,0,0,0,
							ky_pad,ky_pad,kx_pad,kx_pad,
							0,0,0,0),
							mode='constant')
					input_Ek = training_Ek.data[ batch_start_idx : batch_end_idx ]

					output_I = torch.zeros( ( number_in_batch, p.simulation['num_orders'],
												p.data['img_dimension'], p.data['img_dimension'] 
												),
										dtype = torch.float32)
					for batch_idx in range( 0, number_in_batch ):
						for order_idx in range( 0, p.simulation['num_orders'] ):
							# IFFT the Ek appropriately
							# output_E = weight_by_order[ order_idx] * np.fft.ifft2( np.fft.ifftshift( input_Ek[ batch_idx ] * input_tmap[ 0, 0, :, :, 1, order_idx ] ) )
							output_E = weight_by_order[ order_idx] * utility.torch_2dift( input_Ek[ batch_idx ] * input_tmap[ 0, 0, :, :, 1, order_idx ] )
							# Reminder: Transmission idxs are freq., polarization, kx, ky, polarization out, order
							output_I[ batch_idx, order_idx ] = torch.abs( output_E )**2
	 
					# Pull up the ground truth
					ground_truth_output = torch.squeeze( ground_truth_training[ batch_start_idx : batch_end_idx ] )
					ground_truth_output_I = torch.stack([target_intensity_distributions[gt] for gt in ground_truth_output])

					# Zero the gradients accumulated from the previous steps.
					#  If you don’t zero the gradient then you’ll accumulate gradients across multiple batches 
	 				# and over multiple epochs. 
					#  That will mess up your backpropagation and lead to erroneous weight updates.
					nn_optimizer.zero_grad()
	 
					# Run data through our current model, and calculate loss function for evaluation

					nn_model_output = torch.squeeze( nn_model( output_I ) )
					eval_loss = nn_loss_fn( nn_model_output, ground_truth_output_I )

					# Perform backpropagation, and update model parameters
					eval_loss.backward()
					nn_optimizer.step()
	 
					average_loss += ( eval_loss.item() / number_batches )
					# print( eval_loss.item() )

				log_file.write( 'Average loss function for epoch ' + str( nn_epoch ) + ' is ' + str( average_loss ) + "\n" )
				print( f'Average loss function for epoch {nn_epoch} is {average_loss:.6f}' )
	
			log_file.close()
   
			#! It's been awhile, but we are still in the Qvalue-loop for the same value of Q_abs.

			#* Evaluation Loop for this value of Q_abs
			def eval_model_by_dataset( dataset_Ek_, ground_truth_ ):
				'''Evaluates loss/error of the current model for each dataset.
				Inputs: Ek dataset and corresponding ground truth
	   			Outputs: Evaluation loss'''
		  
				# Put model into evaluation mode for computing loss and accuracies
				nn_model.eval()

				# TODO: Really should outsource this to some other function instead of rewriting it multiple times in this file.
				# Now we pull up data for the NN: the output_I, calculated from
				# - the input_tmap (we got it from running evaluate_k_space() again)
				# - the input_Ek
				input_tmap = np.pad(
						transmission_map_device_,
						(
							( 0, 0 ), ( 0, 0 ),
							( kx_pad, kx_pad ),
							( ky_pad, ky_pad ),
							( 0, 0 ), ( 0, 0 ) ),
						mode='constant' )
				input_Ek = dataset_Ek_

				output_I = np.zeros( ( len( dataset_Ek_ ), p.simulation['num_orders'],
												p.data['img_dimension'], p.data['img_dimension'] 
												),
										dtype = np.float32)
				for batch_idx in range( 0, len( dataset_Ek_ ) ):
					for order_idx in range( 0,  p.simulation['num_orders'] ):
						output_E = weight_by_order[ order_idx ] * utility.torch_2dift( input_Ek[ batch_idx ][0] * input_tmap[ 0, 0, :, :, 1, order_idx ] )
						output_I[ batch_idx, order_idx ] = np.abs( output_E )**2
				output_I = torch.from_numpy( output_I )
				output_I_sum = torch.sum(output_I, 1)

				# Pull up the ground truth
				ground_truth_output = torch.squeeze( ground_truth_ )
				ground_truth_output_I = torch.stack([target_intensity_distributions[gt] for gt in ground_truth_output])

				# Zero the gradients accumulated from the previous steps.
				#  If you don’t zero the gradient then you’ll accumulate gradients across multiple batches 
				# and over multiple epochs. 
				#  That will mess up your backpropagation and lead to erroneous weight updates.
				nn_optimizer.zero_grad()
	
				# Run data through our current model, and calculate loss function for evaluation

				nn_model_output = torch.squeeze( nn_model( output_I_sum ) )
				eval_loss = nn_loss_fn( nn_model_output, ground_truth_output_I )
    
				#! TODO: Move this to a proper test accuracy code block.
    			# Look at the output intensity and integrate within all the detection zones. Which one has the most intensity?
				with torch.no_grad():			
					output_I_detection_zones = np.zeros( ( len(dataset_Ek_), len(dataset_samples.classes) ) )
					for class_idx in range(0, len(dataset_samples.classes)):
						output_I_detection_zones[:, class_idx] = torch.sum(nn_model_output * \
																		target_intensity_distributions[class_idx],
																		dim = (1,2))
					pred = np.argmax(output_I_detection_zones, 1)
					correct = np.equal(pred, ground_truth_.detach().numpy()).sum()
					accuracy = float(100. * correct / len(ground_truth_))

				return eval_loss.item(), accuracy
			
			training_error[ Q_idx ], _ = eval_model_by_dataset( training_Ek, ground_truth_training )
			print(f'Evaluating Training Error for Q_abs={Qabs}: {training_error[Q_idx]:.5f}')
			validation_error[ Q_idx ], _ = eval_model_by_dataset( validation_Ek, ground_truth_validation )
			print(f'Evaluating Validation Error for Q_abs={Qabs}: {validation_error[Q_idx]:.5f}')
			test_error[ Q_idx ], test_accuracy[ Q_idx ] = eval_model_by_dataset( validation_Ek, ground_truth_validation )
			#! TODO: Set the inputs to the right datasets. Right now it's repeated because of debugging ease.
			# test_error[ Q_idx ], test_accuracy[ Q_idx ] = eval_model_by_dataset( test_Ek, ground_truth_test )

			# Put model into evaluation mode for computing loss and accuracies
			nn_model.eval()
   

			#* Another photonic optimization uses this function to get the evaluation loss and then minimize it.
	
			def function_nlopt(x, gradn):
				'''Calculates loss in the same way as the evaluation loss is calculated in eval_model_by_dataset.
				In this case, this function is meant to be used in NLOpt or a simple gradient descent algorithm.
				Inputs: x = optimization_variable i.e. permittivity block values;
						gradn = initial seed gradient. Unused except to provide a shape. Ultimately overwritten.
	   			Outputs: eval_loss. The gradn passed as input is also overwritten at a higher scope.'''
		  
				log_file = open( p.DATA_FOLDER + "/log.txt", 'a' )
				start_time = time.time()

				global iteration_counter_diffraction
				global dynamic_weights
				global transmission_map_device_
				global photonic_batch_idx_diffraction
				print(f'Overall iteration_counter_diffraction is {iteration_counter_diffraction} and Photonic Batch Idx_Diffraction is {photonic_batch_idx_diffraction}.')
	
				batch_start_idx = training_batch_start_idxs[ photonic_batch_idx_diffraction ]
				batch_end_idx = training_batch_end_idxs[ photonic_batch_idx_diffraction ]
				number_in_batch = batch_end_idx - batch_start_idx
	

				# We need the output intensity AFTER the metaoptic. (Note: ultimately we need a scalar)
				# This is calculated from:
				# the input E_k, and the transmission map obtained by evaluate_k_space(). 

				# Create a Torch Tensor from the transmission map 
				# Padded out as necessary because RCWA only uses the k-values that are defined by the NA (the theta-value is input)
				# So you pad it out as necessary
				input_tmap = torch.from_numpy(
					np.pad(
						transmission_map_device_,
						(
							( 0, 0 ), ( 0, 0 ),
							( kx_pad, kx_pad ),
							( ky_pad, ky_pad ),
							( 0, 0 ), ( 0, 0 ) ),
						mode='constant' ) )
				input_tmap.requires_grad = True
				# Create a Torch Tensor from the Ek map that corresponds to this particular transmission map
				input_Ek = training_Ek[ batch_start_idx : batch_end_idx ][0]

				output_I = torch.zeros( ( number_in_batch, p.simulation['num_orders'],
											p.data['img_dimension'], p.data['img_dimension'] ) )
				for batch_idx in range( 0, number_in_batch ):
					for order_idx in range( 0, p.simulation['num_orders'] ):
						# IFFT the Ek appropriately
						output_E = weight_by_order[order_idx] * \
									utility.torch_2dift(input_Ek[ batch_idx ][0] *
							 							input_tmap[ 0, 0, :, :, 1, order_idx ]
														)
						# Reminder: Transmission idxs are freq., polarization, kx, ky, polarization out, order
						output_I[ batch_idx, order_idx ] = torch.abs( output_E )**2
				output_I_sum = torch.sum(output_I, 1)

	
				# Pull up the ground truth
				ground_truth_output = torch.squeeze( ground_truth_training[ batch_start_idx : batch_end_idx ] )
				ground_truth_output_I = torch.stack([target_intensity_distributions[gt] for gt in ground_truth_output])

				#! We are not zero-gradding here! We WANT the gradient.
	
				# Run data through our current model, and calculate loss function for evaluation

				nn_model_output = torch.squeeze( nn_model( output_I_sum ) )
				eval_loss = nn_loss_fn( nn_model_output, ground_truth_output_I )

				# Perform backpropagation, and update model parameters
				eval_loss.backward()
	

				#*-------------------------------------------------------------------------------------------------
	
				# Obtain gradient of total_intensity with respect to the transmission map.
				tmap_gradient = input_tmap.grad.detach().numpy()

				# Unpad - Extract the portion of this that corresponds to transmission_map_device_
				mid_grad2_x = tmap_gradient.shape[ 2 ] // 2
				mid_grad2_y = tmap_gradient.shape[ 3 ] // 2
				tmap_size_x = transmission_map_device_.shape[ 2 ] 
				tmap_size_y = transmission_map_device_.shape[ 3 ]
				tmap_offset_x = tmap_size_x // 2
				tmap_offset_y = tmap_size_y // 2

				extract_tmap_gradient = tmap_gradient[
									:, :,
									( mid_grad2_x - tmap_offset_x ) : ( mid_grad2_x - tmap_offset_x + tmap_size_x ),
									( mid_grad2_y - tmap_offset_y ) : ( mid_grad2_y - tmap_offset_y + tmap_size_y ),
									:, : 
								]
	
				#* Use current gradient of evaluation loss w.r.t. transmission map to obtain updated gradient and FoM
				# Above, we ran evaluate_k_space() for the transmission_map_device as obtained from one simulation through RCWA.
				# We didn't use that obtained gradient at all - it didn't correlate to the current gradient at all.
				# Here, we run it now to obtain the updated gradient for optimization (whether through NLOpt, PyTorch, or whatever).
				# And the input gradient is the gradient of the evaluation loss w.r.t. the transmission map as obtained by PyTorch.
				# i.e. current optimization gradient.
				# figure_of_merit_, gradn[ : ], figure_of_merit_individual_, transmission_map_device_ = evaluate_k_space(
				# 	Qabs, x, extract_tmap_gradient )
	
				gradn[ : ] = extract_tmap_gradient
	
				# print( extract_tmap_gradient.shape )
				# print(  np.max(np.abs(extract_tmap_gradient)))
				# print(  np.max(np.abs(tmap_gradient)))
				# print( np.max(np.abs(transmission_map_device_)))
				# print(np.max(np.abs(othergrad)))
	
				log_file.write(
					'Q step = ' + str( Q_idx ) +
					' out of ' + str( num_Q_iterations - 1 ) +
					', Optimization step = ' + str( iteration_counter ) +
					 ', and loss = ' + str( eval_loss.item() ) + '\n' )
				print( 'Q step = ', Q_idx, ' out of ', ( num_Q_iterations - 1 ), ', Optimization step = ', iteration_counter, ', and loss = ', eval_loss.item() )
	
				# Update counters #! which do these correspond to?
				iteration_counter_diffraction += 1
				photonic_batch_idx_diffraction = ( photonic_batch_idx_diffraction + 1 ) % number_batches

				elapsed_time = time.time() - start_time
				log_file.write(
					'It took ' + str( elapsed_time ) + ' seconds to run one iteration!\n' )
				log_file.close()

				# Return evaluation loss as scalar
				return eval_loss.item()
   
			
			log_file = open( p.DATA_FOLDER + "/log.txt", 'a' )

			# Set up an NLopt instance for any MMA-based optimization that might be chosen or enabled
			optimization = nlopt.opt( nlopt.LD_MMA, total_dof )
			optimization.set_lower_bounds( lower_bounds )
			optimization.set_upper_bounds( upper_bounds )

			# Set relative tolerance on optimization parameters.
			optimization.set_xtol_rel( p.optimization['optimization_tolerance'] )
			# Stop when the number of function evaluations exceeds maxeval. (0 or negative for no limit.)
			optimization.set_maxeval( p.optimization['num_photonic_grad_descent_iter'] )
			# Set objective function
			optimization.set_min_objective( function_nlopt )

			# last_fom = 0
			# num_photonic_grad_descent_iter = number_batches
   
			# If MMA is desired, calls NLopt as the optimizer for diffraction efficiency.
			if p.optimization['enable_method_of_moving_asymptotes']:
				# optimization_variable = optimization.optimize( optimization_variable )
				transmission_map_device_ = optimization.optimize( transmission_map_device_ )
			if not p.optimization['enable_method_of_moving_asymptotes']:
				for idx in range( 0, p.optimization['num_photonic_grad_descent_iter'] ):
					print(f'Photonic grad descent idx: {idx}')
		
					# gradn = np.zeros( optimization_variable.shape )
					gradn = np.zeros(transmission_map_device_.shape)

					fom = function_nlopt( transmission_map_device_, gradn )
					normalized_gradn = gradn / np.max( np.abs( gradn ) )

					log_file.write(
						'Q step = ' + str( Q_idx ) +
						' out of ' + str( num_Q_iterations - 1 ) +
						', Optimization step = ' + str( idx ) +
						 ', and loss = ' + str( fom ) + '\n' )

					# optimization_variable -= p.optimization['gradient_descent_step_size_normalized'] * normalized_gradn
					# optimization_variable = np.minimum( device_permittivity, np.maximum( device_background_permittivity, optimization_variable ) )
					transmission_map_device_ -= p.optimization['gradient_descent_step_size_normalized'] * normalized_gradn
					transmission_map_device_ = np.minimum( 1.0, np.maximum( 0.0, transmission_map_device_ ) )
			
			
   			# figure_of_merit_, gradn_, figure_of_merit_individual, transmission_map_device_ = evaluate_k_space( Qabs, optimization_variable )		
   
			log_file.close()
   
			np.save( p.DATA_FOLDER + '/optimization_variable_' + str( Q_idx ) + '.npy', optimization_variable )
			np.save( p.DATA_FOLDER + '/transmission_map_' + str( Q_idx ) + '.npy', transmission_map_device_ )

			np.save( p.DATA_FOLDER + '/training_error.npy', training_error )
			np.save( p.DATA_FOLDER + '/validation_error.npy', validation_error )
			np.save( p.DATA_FOLDER + '/test_error.npy', test_error )
			np.save( p.DATA_FOLDER + '/test_accuracy.npy', test_accuracy )

			if ( Q_idx == ( num_Q_iterations - 1 ) ):
				np.save( p.DATA_FOLDER + '/optimization_variable_final.npy', optimization_variable )
				np.save( p.DATA_FOLDER + '/transmission_map_final.npy', transmission_map_device_ )
				torch.save( nn_model.state_dict(), p.DATA_FOLDER + '/final_model.pt' )

		#* We have reached the end of the Q-loop.
		terminate_message = { "instruction" : "terminate" }
		for processor_rank_idx_ in range( 1, resource_size ):
			comm.send( terminate_message, dest=processor_rank_idx_ )
   
		print( "Time Ended: " + datetime.now().strftime(r"%d/%m/%Y %H:%M:%S"))
		print(f'Program Timer: {(time.time() - program_starttime):.1f}' + "\n")
   

	elif run_mode == 'eval':
		# TODO: Eventually the below piece of code should be moved to a separate evaluate.py file. 
  		# TODO: Figure out a way to pass everything from above into a new file without losing anything.
	 
		# Define overall folder names and structures
		p.DATA_FOLDER = override_data_folder
		p.PLOT_FOLDER = os.path.join(override_data_folder, 'plots')
		if not os.path.isdir(p.PLOT_FOLDER):
			os.mkdir(p.PLOT_FOLDER)
   
		# Reload all saved variables
		optimization_variable = np.load( p.DATA_FOLDER + '/optimization_variable_final.npy' )
		# transmission_map_device_ = np.load( p.DATA_FOLDER + '/transmission_map_final.npy' )
		transmission_map_device_ = np.load( p.DATA_FOLDER + '/transmission_map_4.npy' )
		# transmission_map_device_ = np.ones( desired_transmission.shape, dtype=complex )
		training_error = np.load(p.DATA_FOLDER + '/training_error.npy')
		validation_error = np.load(p.DATA_FOLDER + '/validation_error.npy')
		test_error = np.load(p.DATA_FOLDER + '/test_error.npy')
		test_accuracy = np.load(p.DATA_FOLDER + '/test_accuracy.npy')
  
		#* Load up NN model from before
		nn_model = networks.identity_net(p.simulation['num_orders'])
		# nn_model = networks.simple_net(p.simulation['num_orders'])
		nn_model.load_state_dict( torch.load( p.DATA_FOLDER + '/final_model.pt' ) )
		# Define loss function
		nn_loss_fn = torch.nn.MSELoss()
  
		# #* Visualize optimization variable (permittivity)

		# optimized_permittivity = np.reshape( optimization_variable, ( num_design_layers * Nx, Ny ) )
		# plt.subplot( 1, 2, 1 )
		# plt.imshow( optimized_permittivity )
		# plt.colorbar()
		# plt.subplot( 1, 2, 2 )
		# plt.imshow( np.reshape( init_optimization_variable, ( num_design_layers * Nx, Ny ) ) )
		# plt.colorbar()
		# plt.show()


		#* Visualize output intensity for an image when passed through the transmission map	 
		output_I_folder = 'output_intensity'
		p.INTENSITY_FOLDER = os.path.join(p.PLOT_FOLDER, output_I_folder)
		if not os.path.isdir(p.INTENSITY_FOLDER):
			os.makedirs(p.INTENSITY_FOLDER)
		file_type_name = 'output_I'
  
		def eval_model_plot_by_dataset( dataset_Ek_, ground_truth_ ):
			'''Evaluates loss/error of the current model for each dataset.
			Inputs: Ek dataset and corresponding ground truth
			Outputs: Evaluation loss'''
   
			global p
			global file_type_name
			global transmission_map_device_
	  
			# Put model into evaluation mode for computing loss and accuracies
			nn_model.eval()
		
			# TODO: Really should outsource this to some other function instead of rewriting it multiple times in this file.
			# Now we pull up data for the NN: the output_I, calculated from
			# - the input_tmap (we got it from running evaluate_k_space() again)
			# - the input_Ek
			input_tmap = np.pad(
					transmission_map_device_,
					(
						( 0, 0 ), ( 0, 0 ),
						( kx_pad, kx_pad ),
						( ky_pad, ky_pad ),
						( 0, 0 ), ( 0, 0 ) ),
					mode='constant' )
			input_Ek = dataset_Ek_

			output_I = np.zeros( ( len( dataset_Ek_ ), p.simulation['num_orders'], 
									p.data['img_dimension'], p.data['img_dimension']
									), dtype=np.float32 
								)
			for batch_idx in range( 0, len( dataset_Ek_ ) ):
				for order_idx in range( 0, p.simulation['num_orders'] ):
					output_E = weight_by_order[ order_idx ] * utility.torch_2dift( input_Ek[ batch_idx ] * input_tmap[ 0, 0, :, :, 1, order_idx ] )
					output_I[ batch_idx, order_idx ] = np.abs( output_E )**2
			output_I = torch.from_numpy( output_I )
			output_I_sum = torch.sum(output_I, 1)

			# Pull up the ground truth
			if len(ground_truth_.size()) <= 0:
				ground_truth_output = torch.unsqueeze( ground_truth_, 0)
				ground_truth_ = torch.unsqueeze( ground_truth_, 0)
			else:
				ground_truth_output = torch.squeeze( ground_truth_ )

			ground_truth_output_I = torch.stack([target_intensity_distributions[gt] for gt in ground_truth_output])

			# Run data through our current model, and calculate loss function for evaluation
			nn_model_output = torch.squeeze( nn_model( output_I_sum ) )

			#* Plot the intensity distribution for each order, and the sum over all orders i.e. output intensity at detector plane
			print('order up')
			plt.subplot(2, 3, 1)
			plt.imshow( torch.sum(output_I,1)[0] )
			plt.title('Sum over Orders')
			plt.colorbar()
			for order_idx in range( 1, p.simulation['num_orders'] ):
				plt.subplot( 2, 3, order_idx + 1 )
				plt.imshow( output_I[ 0, order_idx-1 ] )
				plt.title(f'Order {order_idx-1}')
				plt.colorbar()
			# plt.show()
			plt.savefig(f'{p.INTENSITY_FOLDER}/{file_type_name}_orders.png',
					bbox_inches='tight')
			plt.close()

			#* Plot the output intensity after passing through the NN and the target intensity distribution
			plt.subplot( 1, 2, 1 )
			plt.imshow( nn_model_output.detach().numpy()[ 0 ] )
			# plt.imshow( nn_model_output.detach().numpy() )
			plt.title('(NN) Output Intensity')
			plt.colorbar()
			plt.subplot( 1, 2, 2 )
			plt.imshow( target_intensity_distributions[6].detach().numpy() )
			# plt.imshow( ground_truth_output_I[ 0 ].detach().numpy() )
			plt.title('Target Intensity Distribution')
			plt.colorbar()
			# plt.show()
			# plt.imshow( np.abs( nn_model_output.detach().numpy()[ 0 ] - ground_truth_[ 0 ] )**2 )
			# plt.colorbar()
			# plt.show()
			plt.savefig(f'{p.INTENSITY_FOLDER}/{file_type_name}_comparison.png',
					bbox_inches='tight')
			plt.close()

			# return eval_loss.item()

			# # Produce 2D cuts through both the NN model output intensity and the ground truth output intensity
			# axis_idx = ground_truth_output_I[0].size()[0] // 2
			# plt.subplot( 1, 2, 1 )
			# # plt.plot( nn_model_output.detach().numpy()[ 0 ][ 51, : ] )
			# plt.plot( nn_model_output.detach().numpy()[ axis_idx, : ] )
			# plt.plot( ground_truth_output_I[ 0 ][ axis_idx, : ], color='g' )
			# plt.show()
			# plt.subplot( 1, 2, 2 )
			# # plt.plot( nn_model_output.detach().numpy()[ 0 ][ :, 51 ] )
			# plt.plot( nn_model_output.detach().numpy()[ :, axis_idx ] )
			# plt.plot( ground_truth_output_I[ 0 ][ :, axis_idx ], color='g' )
			# plt.show()
   
			print('All plots for dataset generated.')
	 
		dataset_sample_idx = 58
		# Show original image and FT
		plt.subplot( 1, 3, 1 )
		im1 = plt.imshow(torch.squeeze(training_samples[dataset_sample_idx][0]))
		# plt.colorbar()
		plt.title('Image')
		# Fourier Transform
		plt.subplot( 1, 3, 2 )
		im2 = plt.imshow(np.real(torch.squeeze(training_Ek[dataset_sample_idx][0]).detach().numpy()))
		# plt.colorbar()
		plt.title('FT')
		plt.savefig(f'{p.INTENSITY_FOLDER}/image_and_FT.png',
					bbox_inches='tight')
		plt.close()
  
		eval_model_plot_by_dataset( validation_Ek.data[ dataset_sample_idx:dataset_sample_idx+3 ], 
                             			ground_truth_validation[ dataset_sample_idx:dataset_sample_idx+3 ] )
		eval_model_plot_by_dataset( training_Ek.data[ dataset_sample_idx:dataset_sample_idx+3 ], 
                             			ground_truth_training[ dataset_sample_idx:dataset_sample_idx+3 ] )
		
		
		# #* Visualize target zones
		# utility.target_classes_to_zones(dataset_samples.classes, p.data['img_dimension'])
	
	
		#* Visualize training and validation error of model at each value of Qabs
		# Reminder that the error is defined as the evaluation loss between the output intensity as calculated, and the ground truth.

		file_type_name = 'tr_val_err'

		# Regenerate the Q-values.
		# Qabs is a parameter for relaxation to better approach global optimal, at Qabs = inf, it will describe the real physics.
		# It can also be used to resolve any singular matrix issues by setting a large but finite Qabs, e.g. Qabs = 1e5.
		Q_values_optimization = None
		if p.optimization['Q_ramp_style'] == 'linear':
			Q_values_optimization = utility.Q_ramp_linear(  p.simulation['Q_ramp_start'],
															p.simulation['Q_ramp_end'],
															p.simulation['Q_number_steps'] )
		elif p.optimization['Q_ramp_style'] == 'exponential':
			Q_values_optimization = utility.Q_ramp_exponential( p.optimization['Q_ramp_start'],
																p.optimization['Q_ramp_end'],
																p.optimization['Q_number_steps'] )
		else:
			print( 'Unknown Q ramp style!' )
			sys.exit( 1 )
		# We will always end on Q=inf to simulate real life e.g. aperiodic context
		Q_values_optimization = np.append(Q_values_optimization, np.inf)
		num_Q_iterations = len(Q_values_optimization)

		print(r'Plotting training and validation error over different values of Q_abs.')

		plt.plot(range(0, num_Q_iterations), training_error, '.-', label='Training Error')
		plt.plot(range(0, num_Q_iterations), validation_error, '.-', label='Validation Error')

		plt.legend()
		plt.title(f'Training and Validation Error')
		plt.xlabel('Q-Iteration')
		plt.savefig(f'{p.PLOT_FOLDER}/{file_type_name}.png',
					bbox_inches='tight')
		plt.close()
  
		#* Visualize test accuracy of model at each value of Qabs

		file_type_name = 'test_acc'
		print(r'Plotting test accuracy over different values of Q_abs.')

		plt.plot(range(0, num_Q_iterations), test_accuracy, '.-', color='black', label='Test Accuracy')

		plt.legend()
		plt.title(f'Test Accuracy')
		plt.xlabel('Q-Iteration')
		# plt.ylim([0, 100])
		plt.savefig(f'{p.PLOT_FOLDER}/{file_type_name}.png',
					bbox_inches='tight')
		plt.close()
  
  
		#* Visualize transmission maps of device
		transmission_maps_folder = 'transmission_maps'
		p.TRANSMISSION_FOLDER = os.path.join(p.PLOT_FOLDER, transmission_maps_folder)
		if not os.path.isdir(p.TRANSMISSION_FOLDER):
			os.makedirs(p.TRANSMISSION_FOLDER)
		file_type_name = 'tk_map'
  
		final_device_flag = False
		# Create transmission maps for every polarization_out and order, for every single device produced at each Q_abs.
		# Additionally store a second copy of the maps for the final device in its own dedicated folder.
		for Q_idx in range(0, p.optimization['Q_number_steps']):
			try:
				transmission_map_device_plot = np.load( p.DATA_FOLDER + f'/transmission_map_{Q_idx}.npy' )
			except Exception as ex:
				transmission_map_device_plot = np.load( p.DATA_FOLDER + f'/transmission_map_final.npy' )
				final_device_flag = True
			if Q_idx == p.optimization['Q_number_steps']-1:
				final_device_flag = True

			for polarization_idx in range(0, p.simulation['num_polarizations_out']):
				for order_idx in range(0, p.simulation['num_orders']):
					print(f'Plotting transmission map t(k) for Qidx {Q_idx}: pol. {polarization_idx}, order {order_idx}...')
		
					# Reminder: Transmission idxs are freq., polarization, kx, ky, polarization out, order
					# plt.imshow(np.real(transmission_map_device_plot[0,0,:,:, polarization_idx, order_idx]),
					# 			cmap = plt.get_cmap('jet'),
					# 			vmin=0, vmax=1)
					plt.imshow(np.real(transmission_map_device_plot[0,0,:,:, polarization_idx, order_idx]))
     
					# from matplotlib import cm
					# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
					# KX, KY = np.meshgrid(kx_values, ky_values)
					# surf = ax.plot_surface(KX, KY, np.real(transmission_map_device_plot[0,0,:,:, polarization_idx, order_idx]), 
					# 					cmap=cm.coolwarm,
					# 					linewidth=0, antialiased=False)
		
					plt.colorbar()
					plt.title(f'Transmission Map - pol {polarization_idx}, order {order_idx}')
					plt.xlabel('k_x')
					plt.ylabel('k_y')
					plt.savefig(f'{p.TRANSMISSION_FOLDER}/{file_type_name}_Q{Q_idx}_{polarization_idx}_{order_idx}.png',
								bbox_inches='tight')
					if final_device_flag:
						if not os.path.isdir(p.TRANSMISSION_FOLDER + '/final'):
							os.makedirs(p.TRANSMISSION_FOLDER + '/final')
						plt.savefig(f'{p.TRANSMISSION_FOLDER}/final/{file_type_name}_{polarization_idx}_{order_idx}.png',
								bbox_inches='tight')
					plt.close()
     
				
  		# TODO: Check gradients

#* If not the master, then this is one of the workers!

else:

	# Run a loop waiting for something to do or the notification that the optimization is done
	while True:

		# This is a blocking receive call, expecting information to be sent from the master
		data = comm.recv( source=0 )

		if data[ "instruction" ] == "terminate":
			break
		elif data[ "instruction" ] == "simulate":
			processor_simulate( data )
		else:
			print( 'Unknown call to master ' + str( processor_rank ) + ' of ' + data[ 'instruction' ] + '\nExiting...' )
			sys.exit( 1 )


print("Reached end of file. Program completed.")