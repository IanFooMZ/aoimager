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

# System, Plotting, etc.
import os
import pickle
import sys
import time
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

print("All modules loaded.")
sys.stdout.flush()


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



#* Read in all the parameters! (Hard-coded)

random_seed = parameters[ "random_seed" ]
np.random.random( random_seed )

data_folder = parameters[ "data_folder" ] + "/"
if should_override_data_folder:
	data_folder = override_data_folder

lattice_x = parameters[ "lattice_x" ]
lattice_y  = parameters[ "lattice_y" ]
num_planewaves = parameters[ "num_planewaves" ]

desired_transmission_filepath =parameters.get( "desired_transmission_filename" )
if desired_transmission_filepath is not None:
	desired_transmission_filepath = data_folder + desired_transmission_filepath
static_weights_filepath = parameters.get( "static_weights_filepath" )
if static_weights_filepath is not None:
	static_weights_filepath = data_folder + static_weights_filepath

static_weights_specifier = parameters.get( "static_weights_specifier" )
dynamic_weights_specifier = parameters.get( "dynamic_weights_specifier" )

ratio_normal_to_final = parameters.get( "ratio_normal_to_final" )
decay_theta = parameters.get( "decay_theta" )

Q_ramp_start = parameters[ "Q_ramp_start" ]
Q_ramp_end = parameters[ "Q_ramp_end" ]
Q_ramp_style = parameters[ "Q_ramp_style" ]
Q_number_steps = parameters[ "Q_number_steps" ]
max_iterations_per_Q = parameters[ "max_iterations_per_Q" ]


optimization_tolerance = parameters[ "optimization_tolerance" ]


input_layer_thickness = parameters[ "input_layer_thickness" ]
output_layer_thickness = parameters[ "output_layer_thickness" ]
input_index = parameters[ "input_index" ]
output_index = parameters[ "output_index" ]
design_layer_thicknesses = parameters[ "design_layer_thicknesses" ]
has_spacers = parameters[ "has_spacers" ]
spacer_layer_thicknesses = parameters.get( "spacer_layer_thicknesses" )
spacer_layer_indices = parameters.get( "spacer_layer_indices" )


device_index = parameters[ "device_index" ]
device_background_index = parameters[ "device_background_index" ]


normalize_to_Fresnel = parameters[ "normalize_to_Fresnel" ]


device_initialization = parameters[ 'device_initialization' ]
initialization_blur_sigma = parameters.get( 'initialization_blur_sigma' )
initialization_mean = parameters.get( 'initialization_mean' )
initialization_std = parameters.get( 'initialization_std' )

initiailization_uniform_density = parameters.get( 'initiailization_uniform_density' )



number_additional_dof = parameters[ "number_additional_dof" ]
fom_adjustment_function = parameters.get( "fom_adjustment_function" )

init_function_additional_dof = parameters.get( "init_function_additional_dof" )
bounds_function_additional_dof = parameters.get( "bounds_function_additional_dof" )

permittivity_symmetry_function = parameters.get( "permittivity_symmetry_function" )

standard_optimization_goal = parameters.get( "standard_optimization_goal" )

quadratic_max_transmission = parameters.get( "quadratic_max_transmission" )

phase_shift_max_transmission = parameters.get( "phase_shift_max_transmission" )
phase_imaging_shift_amount = parameters.get( "phase_imaging_shift_amount" )
phase_imaging_orders = parameters.get( "phase_imaging_orders" )


num_frequencies = parameters[ "num_frequencies" ]
num_polarizations = parameters[ "num_polarizations" ]
num_theta = parameters[ "num_theta" ]
num_phi = parameters[ "num_phi" ]
num_orders = parameters[ "num_orders" ]

assert ( num_frequencies == 1 ), "For now we are assuming there is just one frequency!"
assert ( num_polarizations == 1 ), "For now we are assuming there is just one polarization!"

min_frequency = parameters[ "min_frequency" ]
max_frequency = parameters[ "max_frequency" ]

min_theta = parameters[ "min_theta" ]
max_theta = parameters[ "max_theta" ]

polarizations = parameters[ "polarizations" ]

min_phi = parameters[ "min_phi" ]#10 * np.pi / 180.#parameters[ "min_phi" ]
max_phi = parameters[ "max_phi" ]#170. * np.pi / 180.#parameters[ "max_phi" ]



# todo: save datasets just in case the random seed for generation is not compatible across
# the cluster to here (i.e. - does not create the same dataset)

zero_order_weight = parameters.get( "zero_order_weight" )

gradient_descent_step_size_normalized = parameters.get( "gradient_descent_step_size_normalized" )

enable_method_of_moving_asymptotes = parameters.get( "enable_method_of_moving_asymptotes" )
enable_method_of_moving_asymptotes_diffraction = parameters.get( "enable_method_of_moving_asymptotes_diffraction" )

optimize_diffraction_efficiency = parameters.get( "optimize_diffraction_efficiency" )

number_dataset_samples = parameters[ "number_dataset_samples" ]
number_training_samples = parameters[ "number_training_samples" ]
number_validation_samples = parameters[ "number_validation_samples" ]

training_batch_size = parameters[ "training_batch_size" ]

num_nn_epochs_per_photonic_epoch = parameters[ "num_nn_epochs_per_photonic_epoch" ]
num_photonic_grad_descent_iter = parameters[ "num_photonic_grad_descent_iter" ]

assert ( number_training_samples + number_validation_samples ) <= number_dataset_samples, "Too many samples in training and validation!"


img_mesh = parameters[ "img_mesh" ]
img_dimension = parameters[ "img_dimension" ]
angular_spread_size_radians = parameters[ "angular_spread_size_radians" ]
amplitude_spread_bounds = parameters[ "amplitude_spread_bounds" ]

assert ( img_dimension % 2 ) == 1, "Assuming throughout that we have an odd image dimension!"

print('Parameters read-in completed.')



#* Parameter Processing

if zero_order_weight is None:
	zero_order_weight = 1.0
weight_by_order = np.ones( num_orders, dtype=np.float32 )
weight_by_order[ 0 ] = zero_order_weight

if gradient_descent_step_size_normalized is None:
	gradient_descent_step_size_normalized = 0.01

if enable_method_of_moving_asymptotes is None:
	enable_method_of_moving_asymptotes = False

if enable_method_of_moving_asymptotes_diffraction is None:
	enable_method_of_moving_asymptotes_diffraction = False


if optimize_diffraction_efficiency is None:
	optimize_diffraction_efficiency = False

number_test_samples = number_dataset_samples - number_training_samples - number_validation_samples


num_design_layers = len( design_layer_thicknesses )

device_permittivity = device_index**2
device_background_permittivity = device_background_index**2

num_polarizations_out = 2



frequencies = np.linspace( min_frequency, max_frequency, num_frequencies )
theta_values = np.linspace( min_theta, max_theta, num_theta )
phi_values = np.linspace( min_phi, max_phi, num_phi )

numerical_aperture = np.sin( np.max( theta_values ) )


kr_max, kr_values = utility.kr_max( frequencies[ 0 ], img_mesh, img_dimension, numerical_aperture )
kx_values = kr_values
ky_values = kr_values

num_kx = len( kx_values )
num_ky = len( ky_values )
print( "num kx = " + str( num_kx ) )
print( "num ky = " + str( num_ky ) )

#
# todo: AMPLITUDE, PHASE, POLARIZATION?
#

tp, xy = utility.create_k_space_map( 1. / frequencies[ 0 ], img_mesh, img_dimension, numerical_aperture )
print( 'tp = ' + str( len( tp ) ) )

print("All modules loaded.")
sys.stdout.flush()



#* Job Allocation for Running in Parallel
num_jobs = num_frequencies * num_polarizations * num_kx * num_ky

min_jobs_per_resource = int( np.floor( num_jobs / resource_size ) )

jobs_per_resource = [ min_jobs_per_resource for idx in range( 0, resource_size ) ]

remaining_jobs = num_jobs - min_jobs_per_resource * resource_size

for idx in range( 0, remaining_jobs ):
	jobs_per_resource[ idx ] += 1

frequency_opt_idxs = np.arange( 0, num_frequencies )
polarizations_opt_idxs = np.arange( 0, num_polarizations )
kx_opt_idxs = np.arange( 0, num_kx )
ky_opt_idxs = np.arange( 0, num_ky )

all_opt_idxs = np.meshgrid( frequency_opt_idxs, polarizations_opt_idxs, kx_opt_idxs, ky_opt_idxs )
flatten_opt_idxs = []

for frequency_idx in range( 0, num_frequencies ):
	for polarization_idx in range( 0, num_polarizations ):
		for kx_idx in range( 0, num_kx ):
			for ky_idx in range( 0, num_ky ):

				flatten_opt_idxs.append( ( frequency_idx, polarization_idx, kx_idx, ky_idx ) )

resource_start_idxs = []
resource_start_idx = 0
for job_idx in range( 0, resource_size ):
	resource_start_idxs.append( resource_start_idx )
	resource_start_idx += jobs_per_resource[ job_idx ]

jobs_for_resource_start = resource_start_idxs[ processor_rank ]

# When an optimization loop happens, these are the jobs to run on a given processor.
jobs_for_resource = flatten_opt_idxs[ jobs_for_resource_start : ( jobs_for_resource_start + jobs_per_resource[ processor_rank ] ) ]


#* Define transmission map by:
# frequency, input polarization, theta, phi, output polarization, order

# for now, dummy variable (initialise)
desired_transmission = np.zeros( ( num_frequencies, num_polarizations, num_kx, num_ky, num_polarizations_out, num_orders ), dtype=np.complex )

# desired_transmission = None
# if desired_transmission_filepath is not None:
# 	desired_transmission = np.load( desired_transmission_filepath )

# if standard_optimization_goal is not None:
# 	if standard_optimization_goal == "quadratic":
# 		desired_transmission = utility.quadratic_optimization_goal(
# 			quadratic_max_transmission,
# 			num_frequencies, num_polarizations, num_theta, num_phi, num_orders,
# 			polarizations, theta_values )

# 	elif standard_optimization_goal == "phase_imaging_shift":
# 		desired_transmission = utility.phase_imaging_shift_optimization_goal(
# 			phase_shift_max_transmission, phase_imaging_shift_amount,
# 			num_frequencies, num_polarizations, num_theta, num_phi, num_orders,
# 			frequencies, polarizations, theta_values, phi_values, phase_imaging_orders )

# 	else:
# 		print( 'Unrecognized standard optimization goal' )
# 		sys.exit( 1 )


static_weights = None
if static_weights_filepath is not None:
	static_weights = np.load( static_weights_filepath )


# Default the static weights to uniform

if static_weights_specifier is not None:
	if static_weights_specifier == "uniform":
		static_weights = np.ones( desired_transmission.shape )
	elif static_weights_specifier == "normal_amplification":
		static_weights = utility.normal_amplification(
			ratio_normal_to_final, decay_theta, theta_values, desired_transmission.shape )
	elif static_weights_specifier == "high_angle_amplification":
		static_weights = np.flip( utility.normal_amplification(
			1. / ratio_normal_to_final, decay_theta, theta_values, desired_transmission.shape ), axis=2 )
	else:
		print( 'Unrecognized static weights specifier' )
		sys.stdout.flush()
		sys.exit( 1 )


assert num_frequencies == desired_transmission.shape[ 0 ], 'Unexpected desired_transmission shape (frequency)'
assert num_polarizations == desired_transmission.shape[ 1 ], 'Unexpected desired_transmission shape (polarization)'
assert num_kx == desired_transmission.shape[ 2 ], 'Unexpected desired_transmission shape (theta)'
assert num_ky == desired_transmission.shape[ 3 ], 'Unexpected desired_transmission shape (phi)'
assert num_polarizations_out == desired_transmission.shape[ 4 ], 'Unexpected desired_transmission shape (polarization out)'
assert num_orders == desired_transmission.shape[ 5 ], 'Unexpected desired_transmission shape (order)'

assert num_polarizations_out == 2, "Both output polarizations should be in the optimization goal."


#
# Set blank dynamic weights for first iteration
#
number_fom = np.product( desired_transmission.shape )
dynamic_weights = ( 1. / number_fom ) * np.ones( desired_transmission.shape )

Nx = num_planewaves - 1
Ny = Nx


geometry = DeviceGeometry.DeviceGeometry(
	input_layer_thickness, output_layer_thickness, input_index, output_index,
	[ Nx, Ny ], design_layer_thicknesses, has_spacers, spacer_layer_thicknesses, spacer_layer_indices )


def run_single_simulation(
	frequency_idx_, polarization_idx_, kx_idx_, ky_idx_, Qabs_, device_permittivity_,
	weights_, transmission_map_device_, figure_of_merit_individual_,
	gradient_transmission_polarization_and_order_ ):#,
	# desired_transmission_phase_by_frequency_and_order_ ):
	frequency = frequencies[ frequency_idx_ ]
	frequency_Qabs = frequency * ( 1 + 1j / ( 2 * Qabs_ ) )

	polarization = polarizations[ polarization_idx_ ]
	# theta = theta_values[ theta_idx_ ]
	# phi = phi_values[ phi_idx_ ]

	kx = kx_values[ kx_idx_ ]
	ky = ky_values[ ky_idx_ ]

	#
	# Because c = 1
	#
	k = ( 2 * np.pi * frequencies[ frequency_idx_ ] )

	kz_sq = k**2 - kx**2 - ky**2

	kz = np.sqrt( kz_sq )
	kr = np.sqrt( kx**2 + ky**2 )

	theta = np.arctan( kr / kz )
	phi = np.arctan2( ky, kx )

	planewave = { 'p_amp':0, 's_amp':1, 'p_phase':0, 's_phase':0 }
	if polarization == 'p':
		planewave = { 'p_amp':1, 's_amp':0, 'p_phase':0, 's_phase':0 }

	simulation = grcwa.obj( num_planewaves, lattice_x, lattice_y, frequency_Qabs, theta, phi, verbose=0 )
	geometry.add_layers( simulation )
	simulation.Init_Setup()
	simulation.GridLayer_geteps( device_permittivity_ )

	simulation.MakeExcitationPlanewave(
		planewave['p_amp'],
		planewave['p_phase'],
		planewave['s_amp'],
		planewave['s_phase'],
		order = 0 )

	#
	# get the phase of the input and outputs
	#
	forward_fourier = simulation.GetAmplitudes( simulation.Layer_N - 1, geometry.output_fourier_sampling_offset ) [ 0 ]
	input_fourier = simulation.GetAmplitudes( 0, geometry.input_fourier_sampling_offset ) [ 0 ]

	output_length = len( forward_fourier ) // 2

	get_s_lattice_input_data = input_fourier[ 0 : output_length ]
	get_p_lattice_input_data = input_fourier[ output_length : ]

	get_s_lattice_output_data = forward_fourier[ 0 : output_length ]
	get_p_lattice_output_data = forward_fourier[ output_length : ]


	convert_input_s_data = np.cos( phi ) * get_s_lattice_input_data + np.sin( phi ) * get_p_lattice_input_data
	convert_input_p_data = -np.sin( phi ) * get_s_lattice_input_data + np.cos( phi ) * get_p_lattice_input_data


	convert_output_s_data = np.cos( phi ) * get_s_lattice_output_data + np.sin( phi ) * get_p_lattice_output_data
	convert_output_p_data = -np.sin( phi ) * get_s_lattice_output_data + np.cos( phi ) * get_p_lattice_output_data

	#
	# Note: this is only normalizing based on input polarization and assuming the zeroth order transmission.
	# To account for the full normalization, we need to adjust based on the output theta.  For now, we won't consider this
	# added complexity.
	#

	if polarization == 's':
		normalize_output_s = convert_output_s_data / ( convert_input_s_data[ 0 ] + np.finfo( np.float32 ).eps )
		normalize_output_p = convert_output_p_data / ( convert_input_s_data[ 0 ] + np.finfo( np.float32 ).eps )

		# normalize_output_s *= ( geometry.input_index * np.cos( theta_incident ) / ( geometry.output_index * np.cos( theta_transmitted ) ) )
		# normalize_output_p *= ( geometry.input_index * np.cos( theta_incident ) / ( geometry.output_index * np.cos( theta_transmitted ) ) )

	else:
		normalize_output_s = convert_output_s_data / ( convert_input_p_data[ 0 ] + np.finfo( np.float32 ).eps )
		normalize_output_p = convert_output_p_data / ( convert_input_p_data[ 0 ] + np.finfo( np.float32 ).eps )

		# normalize_output_s *= ( geometry.input_index / geometry.output_index )
		# normalize_output_p *= ( geometry.input_index / geometry.output_index )


	theta_incident = theta
	theta_transmitted = np.arcsin( ( geometry.input_index / geometry.output_index ) * np.sin( theta_incident ) )

	if normalize_to_Fresnel:
		normalize_output_s *= ( geometry.input_index * np.cos( theta_incident ) / ( geometry.output_index * np.cos( theta_transmitted ) ) )
		normalize_output_p *= ( geometry.input_index / geometry.output_index )


	normalize_output_by_polarization = [ normalize_output_s, normalize_output_p ]

	#
	# This should be an array that is of size 2 x N_orders where N_orders is the number of orders being optimized over.
	# We assume the s-polarization output goal is first.
	# 
	optimization_goal_slice = gradient_transmission_polarization_and_order_[ frequency_idx_, polarization_idx_, kx_idx_, ky_idx_ ]
	optimization_weights_slice = weights_[ frequency_idx_, polarization_idx_, kx_idx_, ky_idx_ ]

	# normalize_output_by_polarization[ 0 ][ : ] = np.mean( ( device_permittivity_ - 1.5**2 ) / ( 3.67**2 - 1.5**2 ) )
	# normalize_output_by_polarization[ 1 ][ : ] = np.mean( ( device_permittivity_ - 1.5**2 ) / ( 3.67**2 - 1.5**2 ) )
	# fake_t = 0.1 * np.mean( ( device_permittivity_ - 1.5**2 ) / ( 3.67**2 - 1.5**2 ) ) - 1j * np.mean( ( device_permittivity_ - 1.5**2 ) / ( 3.67**2 - 1.5**2 ) )

	# fake_t = np.mean( ( device_permittivity_**2 ) )


	figure_of_merit = 0.0
	for polarization_out_idx in range( 0, len( normalize_output_by_polarization ) ):
		for order_idx in range( 0, num_orders ):

			# individual_fom = np.abs( normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] - desired_transmission_phase_by_frequency_and_order_[ frequency_idx_, order_idx ] * optimization_goal_slice[ polarization_out_idx, order_idx ] )**2

			# individual_fom = -np.imag( optimization_goal_slice[ polarization_out_idx, order_idx ] * normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] )
			# individual_fom = -np.real( optimization_goal_slice[ polarization_out_idx, order_idx ] * normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] )

			# individual_fom = -(
			# 	np.real( optimization_goal_slice[ polarization_out_idx, order_idx ] * normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] ) -
			# 	np.imag( optimization_goal_slice[ polarization_out_idx, order_idx ] * normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] ) )

			# individual_fom = (
			# 	np.real( optimization_goal_slice[ polarization_out_idx, order_idx ] * fake_t ) +
			# 	np.imag( optimization_goal_slice[ polarization_out_idx, order_idx ] * fake_t ) )

			# individual_fom = np.imag( optimization_goal_slice[ polarization_out_idx, order_idx ] * fake_t )

			# individual_fom = -np.real( optimization_goal_slice[ polarization_out_idx, order_idx ] * 1j * fake_t )

			# individual_fom = np.real( optimization_goal_slice[ polarization_out_idx, order_idx ] * fake_t + np.conj( optimization_goal_slice[ polarization_out_idx, order_idx ] ) * np.conj( fake_t ) )

			real_opt_slice = 0.5 * ( optimization_goal_slice[ polarization_out_idx, order_idx ] + np.conj( optimization_goal_slice[ polarization_out_idx, order_idx ] ) )
			imag_opt_slice = ( 0.5 / 1j ) * ( optimization_goal_slice[ polarization_out_idx, order_idx ] - np.conj( optimization_goal_slice[ polarization_out_idx, order_idx ] ) )
			# individual_fom = np.real( real_opt_slice * np.real( fake_t ) + imag_opt_slice * np.imag( fake_t ) )
			individual_fom = np.real( real_opt_slice * np.real( normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] ) + imag_opt_slice * np.imag( normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] ) )


			# individual_fom = np.real( optimization_goal_slice[ polarization_out_idx, order_idx ] * fake_t )

			# individual_fom = -np.imag( optimization_goal_slice[ polarization_out_idx, order_idx ] * normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] )
			# individual_fom = -np.real( optimization_goal_slice[ polarization_out_idx, order_idx ] * normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] )


			# individual_fom = -(
			# 	-np.real( optimization_goal_slice[ polarization_out_idx, order_idx ] * normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] ) +
			# 	np.imag( optimization_goal_slice[ polarization_out_idx, order_idx ] * normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] ) )



			# individual_fom = 2 * np.real( optimization_goal_slice[ polarization_out_idx, order_idx ] * fake_t )
			# individual_fom = -np.imag( optimization_goal_slice[ polarization_out_idx, order_idx ] * fake_t )
			# individual_fom = 2 * np.abs( optimization_goal_slice[ polarization_out_idx, order_idx ] * fake_t )
			# print( optimization_goal_slice[ polarization_out_idx, order_idx ] * fake_t )

			# individual_fom = 2 * np.real( 1j * optimization_goal_slice[ polarization_out_idx, order_idx ] * normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] )

			# individual_fom = np.abs( optimization_goal_slice[ polarization_out_idx, order_idx ] - normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] )


			# individual_fom = optimization_goal_slice[ polarization_out_idx, order_idx ] * normalize_output_by_polarization[ polarization_out_idx ][ order_idx ]
			# individual_fom = np.abs( normalize_output_by_polarization[ polarization_out_idx ][ order_idx ] )**2

			individual_fom_weighted = (
				optimization_weights_slice[ polarization_out_idx, order_idx ] * individual_fom )


			#
			# I do not like this...
			#
			if isinstance( individual_fom, float ):
				figure_of_merit_individual_[ frequency_idx_, polarization_idx_, kx_idx_, ky_idx_, polarization_out_idx, order_idx ] = individual_fom
			else:
				figure_of_merit_individual_[ frequency_idx_, polarization_idx_, kx_idx_, ky_idx_, polarization_out_idx, order_idx ] = individual_fom._value

			pull_t = normalize_output_by_polarization[ polarization_out_idx ][ order_idx ]
			# pull_t = fake_t
			if isinstance( pull_t, complex ):
				transmission_map_device_[ frequency_idx_, polarization_idx_, kx_idx_, ky_idx_, polarization_out_idx, order_idx ] = pull_t
			else:
				transmission_map_device_[ frequency_idx_, polarization_idx_, kx_idx_, ky_idx_, polarization_out_idx, order_idx ] = pull_t._value

			#
			# Add the weighted fom here so that the autograd picks it up and weights the gradient accordingly.
			#
			figure_of_merit += individual_fom# individual_fom_weighted

	return figure_of_merit

def processor_simulate( data ):
	#
	# If we got a call to simulate, then we must have received some other information:
	# combined weighting function for all figures of merit and the current optimization variable.
	#
	optimization_weights = data[ 'weights' ]
	optimization_variable = data[ 'optimization_variable' ]
	optimization_Qabs = data[ 'Qabs' ]

	gradient_transmission_polarization_and_order = data[ 'gradient_transmission_polarization_and_order' ]

	transmission_map_device = np.zeros( desired_transmission.shape, dtype=np.complex )
	figure_of_merit_individual = np.zeros( desired_transmission.shape )

	def run_jobs_for_resource( optimization_variable_ ):
		figure_of_merit_total = 0.0

		device_permittivity_ = preprocess_optimization_variable( optimization_variable_ )

		for job_idx in range( 0, len( jobs_for_resource ) ):
			frequency_idx, polarization_idx, kx_idx, ky_idx = jobs_for_resource[ job_idx ]

			figure_of_merit_total += run_single_simulation(
				frequency_idx, polarization_idx, kx_idx, ky_idx,
				optimization_Qabs, device_permittivity_,
				optimization_weights,
				transmission_map_device, figure_of_merit_individual,
				gradient_transmission_polarization_and_order )

		return figure_of_merit_total

	run_jobs_for_resource_grad = grad( run_jobs_for_resource )

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


	#
	# Nonblocking send for the case where you are sending from rank 0 to rank 0
	#
	if not ( processor_rank == 0 ):
		comm.isend( results, dest=0 )

	return results

def evaluate_k_space( Qabs_, optimization_variable_, gradient_transmission_polarization_and_order ):
	global dynamic_weights
	global desired_transmission

	figure_of_merit_individual = np.zeros( desired_transmission.shape )
	transmission_map_device = np.zeros( desired_transmission.shape, dtype=np.complex )

	gradn = np.zeros( optimization_variable_.shape, dtype=optimization_variable_.dtype )
	figure_of_merit = 0.0


	optimize_message = {
		"instruction" : "simulate",
		"weights" : dynamic_weights * static_weights,
		"optimization_variable" : np.array( optimization_variable_, dtype=np.float64 ),

		# Need to send out some gradient information
		"gradient_transmission_polarization_and_order" : np.array( gradient_transmission_polarization_and_order, dtype=np.complex ),

		"Qabs" : Qabs
	}

	for processor_rank_idx_ in range( 1, resource_size ):
		#
		# Blcoking send (no sends to the rank 0 processor which would cause a lock-up)
		#
		comm.send( optimize_message, dest=processor_rank_idx_ )


	data = optimize_message

	for processor_rank_idx_ in range( 0, resource_size ):
		if processor_rank_idx_ == 0:
			results = processor_simulate( data )
		else:
			results = comm.recv( source=processor_rank_idx_ )

		if not ( results[ "instruction" ] == "results" ):
			print( "Unexpected message received from processor " + str( processor_rank_idx_ ) + "\nExiting..." )
			sys.exit( 1 )

		gradn[ : ] += results[ "net_grad" ]
		figure_of_merit += results[ "figure_of_merit" ]
		
		figure_of_merit_individual_slice = results[ "figure_of_merit_individual" ]
		transmission_map_device_slice = results[ "transmission_map_device" ]

		jobs_for_other_resource = flatten_opt_idxs[
			resource_start_idxs[ processor_rank_idx_ ] : ( resource_start_idxs[ processor_rank_idx_ ] + jobs_per_resource[ processor_rank_idx_ ] ) ]

		for job_idx in range( 0, len( jobs_for_other_resource ) ):
			frequency_idx, polarization_idx, kx_idx, ky_idx = jobs_for_other_resource[ job_idx ]

			for polarization_out_idx in range( 0, num_polarizations_out ):
				for order_idx in range( 0, num_orders ):

					figure_of_merit_individual[
						frequency_idx, polarization_idx, kx_idx, ky_idx, polarization_out_idx, order_idx ] = figure_of_merit_individual_slice[
							frequency_idx, polarization_idx, kx_idx, ky_idx, polarization_out_idx, order_idx ]

					transmission_map_device[
						frequency_idx, polarization_idx, kx_idx, ky_idx, polarization_out_idx, order_idx ] = transmission_map_device_slice[
							frequency_idx, polarization_idx, kx_idx, ky_idx, polarization_out_idx, order_idx ]

	return figure_of_merit, gradn, figure_of_merit_individual, transmission_map_device

def preprocess_permittivity( device_permittivity ):
	#
	# At some point this should turn into the preprocessing filter chain (i.e. - take a density variable to a permittivity one)
	#
	if permittivity_symmetry_function is not None:
		if permittivity_symmetry_function == "c4_symmetry_explicit":
			device_permittivity = utility.c4_symmetry_explicit( device_permittivity, Nx, Ny, num_design_layers )
		elif permittivity_symmetry_function == "vertical_flip_symmetry_explicit":
			device_permittivity = utility.vertical_flip_symmetry_explicit( device_permittivity, Nx, Ny, num_design_layers )
		elif permittivity_symmetry_function == "woodpile":
			device_permittivity = utility.woodpile( device_permittivity, Nx, Ny, num_design_layers )
		elif permittivity_symmetry_function == "woodpile_and_vertical_flip_symmetry_explicit":
			device_permittivity = utility.vertical_flip_symmetry_explicit(
				utility.woodpile( device_permittivity, Nx, Ny, num_design_layers ), Nx, Ny, num_design_layers )
		else:
			print( 'Unrecognized permittivity symmetry function' )
			sys.exit( 1 )

	return device_permittivity

def process_additional_dof( additional_dof_ ):
	desired_transmission_phase_by_frequency_and_order = np.zeros( ( num_frequencies, num_orders ), dtype=np.complex )

	if fom_adjustment_function is not None:
		if fom_adjustment_function == "modify_transmission_flat_phase":
			desired_transmission_phase_by_frequency_and_order = np.reshape( additional_dof_, ( num_frequencies, num_orders ) )
		else:
			print( 'Unrecognized fom adjustment function' )
			sys.exit( 1 )

	return np.exp( 1j * desired_transmission_phase_by_frequency_and_order )

def preprocess_optimization_variable( optimization_variable_ ):

	device_permittivity = preprocess_permittivity( optimization_variable_ )
	
	return device_permittivity


#* Main Optimization Code Loop
# Things start breaking off by master job and worker jobs for the parallel
# compute.  The master is responsible for handling the optimization and the
# workers are responsible for running simulations given the current optimization
# state and then sending back gradient and figure of merit information to the master.
#

total_dof = num_design_layers * Nx * Ny + number_additional_dof
device_dof_end = num_design_layers * Nx * Ny

#* Let the master job the rank 0

if processor_rank == 0:
	np.random.seed( random_seed )

	#
	# Create the dataset
	#
	dataset_samples = utility.create_random_phase_amplitude_samples(
		1. / frequencies[ 0 ],
		img_mesh, img_dimension,
		angular_spread_size_radians, amplitude_spread_bounds,
		numerical_aperture, number_dataset_samples )

	training_samples = dataset_samples[ 0 : number_training_samples ]
	validation_samples = dataset_samples[ number_training_samples : ( number_training_samples + number_validation_samples ) ]
	test_samples = dataset_samples[ -number_test_samples : ]

	number_batches = ( number_training_samples // training_batch_size ) + ( ( number_training_samples % training_batch_size ) > 0 )

	training_batch_start_idxs = [ training_batch_size * batch_idx for batch_idx in range( 0, number_batches ) ]
	training_batch_end_idxs = [ np.minimum(
		training_batch_start_idxs[ batch_idx ] + training_batch_size,
		number_training_samples ) for batch_idx in range( 0, number_batches ) ]

	kx_start = ( img_dimension // 2 ) - ( num_kx // 2 )
	ky_start = ( img_dimension // 2 ) - ( num_ky // 2 )
	kx_pad = ( img_dimension - num_kx ) // 2
	ky_pad = ( img_dimension - num_ky ) // 2

	dataset_Ek = np.zeros( dataset_samples.shape, dtype=np.complex )

	for dataset_idx in range( 0, number_dataset_samples ):
		E_input = dataset_samples[ dataset_idx ]
		Ek_input = np.fft.fftshift( np.fft.fft2( E_input ) )

		for kx_idx in range( 0, num_kx ):
			for ky_idx in range( 0, num_ky ):

				kx = kx_values[ kx_idx ]
				ky = ky_values[ ky_idx ]

				kr = np.sqrt( kx**2 + ky**2 )
				kr_max = ( 2 * np.pi * frequencies[ 0 ] ) * numerical_aperture

				if kr >= kr_max:
					Ek_input[ kx_idx + kx_start, ky_idx + ky_start ] = 0


		dataset_Ek[ dataset_idx ] = Ek_input


	training_Ek = dataset_Ek[ 0 : number_training_samples ]
	validation_Ek = dataset_Ek[ number_training_samples : ( number_training_samples + number_validation_samples ) ]
	test_Ek = dataset_Ek[ -number_test_samples : ]


	ground_truth = np.zeros( ( number_dataset_samples, img_dimension, img_dimension ), dtype=np.float32 )
	for dataset_idx in range( 0, number_dataset_samples ):
		get_phase = np.angle( dataset_samples[ dataset_idx ] )

		pad_phase_angle = np.pad( get_phase, ( ( 1, 1 ), ( 0, 0 ) ), mode='edge' )
		analytical_phase_grad = ( pad_phase_angle[ 2 :, : ] - pad_phase_angle[ 0 : -2, : ] ) / ( 2 * img_mesh )

		ground_truth[ dataset_idx, :, : ] = analytical_phase_grad


	ground_truth_training = ground_truth[ 0 : number_training_samples ]
	ground_truth_validation = ground_truth[ number_training_samples : ( number_training_samples + number_validation_samples ) ]
	ground_truth_test = ground_truth[ -number_test_samples : ]



	#
	# Optimization variable initialization and bounds
	#
	init_optimization_variable = np.zeros( total_dof )

	lower_bounds = np.ones( total_dof, dtype=float )
	upper_bounds = np.ones( total_dof, dtype=float )

	lower_bounds[ 0 : device_dof_end ] = device_background_permittivity
	upper_bounds[ 0 : device_dof_end ] = device_permittivity


	#
	# Device initiailizations
	#
	init_permittivity = np.zeros( Nx * Ny * num_design_layers )
	for layer_idx in range( 0, num_design_layers ):
		blank_density = np.zeros( ( Nx, Ny ) )

		if device_initialization == 'random':
			init_layer_density = np.random.random( blank_density.shape )
			init_layer_density = gaussian_filter( init_layer_density, sigma=initialization_blur_sigma )

			init_layer_density = init_layer_density - np.mean( init_layer_density )# + initialization_mean
			init_layer_density = init_layer_density * initialization_std / np.std( init_layer_density )
			init_layer_density += initialization_mean
			init_layer_density = np.minimum( 1.0, np.maximum( init_layer_density, 0.0 ) )

		elif device_initialization == 'uniform':
			init_layer_density = initiailization_uniform_density * np.ones( blank_density.shape )
		else:
			print( 'Unrecognized device initialization strategy!' )
			sys.exit( 1 )

		init_layer_permittivity = device_background_permittivity + ( device_permittivity - device_background_permittivity ) * init_layer_density

		init_permittivity[ layer_idx * Nx * Ny : ( layer_idx + 1 ) * Nx * Ny ] = init_layer_permittivity.flatten()

	init_optimization_variable[ 0 : device_dof_end ] = init_permittivity

	#
	# Extra dof initializations - right now we assume all extra degrees of freedom are phases for transmission coefficients.
	#
	if number_additional_dof > 0:
		
		if init_function_additional_dof == 'init_phase_dof':
			init_optimization_variable[ device_dof_end : ] = np.pi
		else:
			print( 'Unrecognized additional dof strategy!' )
			sys.exit( 1 )

		if bounds_function_additional_dof == 'bounds_phase_dof':
			lower_bounds[ device_dof_end : ] = 0
			upper_bounds[ device_dof_end : ] = 2 * np.pi
		else:
			print( 'Unrecognized additional dof strategy!' )
			sys.exit( 1 )



	if mode == 'opt':
		log_file = open( data_folder + "/log.txt", 'w' )
		log_file.write( "Log\n")
		log_file.close()

		Q_values_optimization = None

		if Q_ramp_style == 'linear':
			Q_values_optimization = utility.Q_ramp_linear( Q_ramp_start, Q_ramp_end, Q_number_steps )
		elif Q_ramp_style == 'exponential':
			Q_values_optimization = utility.Q_ramp_exponential( Q_ramp_start, Q_ramp_end, Q_number_steps )
		else:
			print( 'Unknown Q ramp style!' )
			sys.exit( 1 )

		#
		# We will always end on Q=inf to simulate real life!
		#
		Q_values_optimization = list( Q_values_optimization )
		Q_values_optimization.append( np.inf )
		Q_values_optimization = np.array( Q_values_optimization )

		iteration_counter = 0
		iteration_counter_diffraction = 0

		optimization_variable = init_optimization_variable.copy()

		num_Q_iterations = len( Q_values_optimization )
		

		learning_rate = .001
		training_momentum = 0.9


		# class simple_net(torch.nn.Module):
		# 	def __init__(self):
		# 		super(simple_net, self).__init__()

		# 		self.conv1 = torch.nn.Conv2d( num_orders, num_orders, 3, padding=1 )
		# 		self.conv2 = torch.nn.Conv2d( num_orders, num_orders, 3, padding=1 )
		# 		self.conv = torch.nn.Conv2d( num_orders, 1, 3, padding=1 )

		# 	def forward(self, x):
		# 		# x = self.conv1( ( x - 0.0168 ) / 0.0887 )
		# 		# x = self.conv2( x )
		# 		# return self.conv( x )

		# 		return self.conv( ( x - 0.0168 ) / 0.0887 )

		# 		# return ( ( x[ :, 0 ] - 0.0168 ) / 0.0887 )

		#
		# Would be good to increase transmission into higher orders
		# You might practically want them to all be similar brightness
		#




		# nn_model = simple_net()
		nn_model = torch.nn.Sequential(
			torch.nn.BatchNorm2d( num_orders ),
			torch.nn.Conv2d( num_orders, num_orders // 2, 3, padding=1 ),
			torch.nn.BatchNorm2d( num_orders // 2 ),
			torch.nn.ReLU(),
			torch.nn.Conv2d( num_orders // 2, num_orders // 2, 3, padding=1 ),
			torch.nn.BatchNorm2d( num_orders // 2 ),
			torch.nn.ReLU(),
			torch.nn.Conv2d( num_orders // 2, 1, 3, padding=1 )
		)

		#
		# skip connections?
		# phase and amplitude co-recovery?
		#

		# nn_model = torch.nn.Sequential(
		# 	torch.nn.BatchNorm2d( num_orders ),
		# 	torch.nn.Conv2d( num_orders, num_orders, 3, padding=1 ),
		# 	torch.nn.BatchNorm2d( num_orders ),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Conv2d( num_orders, num_orders // 2, 3, padding=1 ),
		# 	torch.nn.BatchNorm2d( num_orders // 2 ),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Conv2d( num_orders // 2, num_orders // 2, 3, padding=1 ),
		# 	torch.nn.BatchNorm2d( num_orders // 2 ),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Conv2d( num_orders // 2, 1, 3, padding=1 )
		# )


		nn_optimizer = torch.optim.SGD( nn_model.parameters(), lr=learning_rate, momentum=training_momentum )
		nn_loss_fn = torch.nn.MSELoss()

		training_error = np.zeros( num_Q_iterations )
		validation_error = np.zeros( num_Q_iterations )


		for Q_idx in range( 0, num_Q_iterations ):
			log_file = open( data_folder + "/log.txt", 'a' )
			log_file.write( "Q idx: " + str( Q_idx ) + " out of " + str( num_Q_iterations - 1 ) + "\n")
			log_file.close()

			Qabs = Q_values_optimization[ Q_idx ]

			#
			# todo: we should also compare to optimizing the network and device at the same time.
			#
			# todo: do we want to also encourage just high transmission? (or we can add some noicse
			# which might naturally increase the amount of transmission)
			#

			#
			# First, let's optimize the device based on the current network.
			# By computing the transmission map of the device into each order, we can
			# compute an average loss function from the network over minibatches of input
			# data.
			# For each minibatch, we get a net gradient for the transmission map wrt the
			# current network.  With that gradient, we can change the device permittivity.
			# If we do this inside the nlopt function and pick a different minibatch each time,
			# time, the optimizer will have essentially a changin gobjctive function each
			# iterations (not sure if this is problematic but something to keep in mind).
			#

			blank_gradient = np.ones( desired_transmission.shape, dtype=np.complex )

			figure_of_merit, gradn, figure_of_merit_individual_, transmission_map_device_ = evaluate_k_space( Qabs, optimization_variable, blank_gradient )

			#
			# Let's put a batch norm on the input intensities because the normalization is going
			# to change by minibatch and current device.
			#

			photonic_batch_idx = 0
			photonic_batch_idx_diffraction = 0



			if optimize_diffraction_efficiency:

				def function_opt_diffraction( x, gradn ):
					log_file = open( data_folder + "/log.txt", 'a' )

					start_time = time.time()
					global iteration_counter_diffraction
					global dynamic_weights
					global transmission_map_device_
					global photonic_batch_idx_diffraction

					batch_start_idx = training_batch_start_idxs[ photonic_batch_idx_diffraction ]
					batch_end_idx = training_batch_end_idxs[ photonic_batch_idx_diffraction ]
					number_in_batch = batch_end_idx - batch_start_idx

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
					input_Ek = training_Ek[ batch_start_idx : batch_end_idx ]
					input_Ek = torch.from_numpy( input_Ek )

					input_I = np.abs( training_samples[ batch_start_idx : batch_end_idx ] )**2
					average_input_I_by_batch = np.squeeze( np.mean( input_I, axis=( 1, 2 ) ) )

					output_I = torch.zeros( ( number_in_batch, num_orders, img_dimension, img_dimension ) )
					for batch_idx in range( 0, number_in_batch ):
						for order_idx in range( 0, num_orders ):
							output_E = torch.fft.ifft2( torch.fft.ifftshift( input_Ek[ batch_idx ] * input_tmap[ 0, 0, :, :, 1, order_idx ] ) )
							output_I[ batch_idx, order_idx ] = torch.abs( output_E )**2


					sum_dimensions = tuple( np.arange( 2, len( output_I.shape ) ) )
					normalized_intensity_by_batch_and_order = torch.mean( output_I, dim=sum_dimensions )
					log_file.write( str( normalized_intensity_by_batch_and_order.shape ) + '\n' )

					detach_zeroth_order = normalized_intensity_by_batch_and_order.detach().numpy()[ :, 0 ]

					normalized_intensity_by_order = torch.zeros( ( number_in_batch, num_orders ) )
					for order_idx in range( 1, num_orders ):
						normalized_intensity_by_order[ :, order_idx ] = (
							normalized_intensity_by_batch_and_order[ :, order_idx ] / torch.from_numpy( detach_zeroth_order ) )
					normalized_intensity_by_order = torch.squeeze( torch.mean( normalized_intensity_by_order, dim=0 ) )
					log_file.write( str( normalized_intensity_by_order.shape ) + '\n' )

					#
					# optimizing for diffracted orders greater than 0th order - do we need to divide out by the order weight?
					#						
					total_intensity = torch.prod( normalized_intensity_by_order[ 1 : ] )
					total_intensity.backward()

					tmap_gradient = input_tmap.grad.detach().numpy()

					# othergrad = output_I.grad.detach().numpy()

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
						:, : ]

					figure_of_merit_, gradn[ : ], figure_of_merit_individual_, transmission_map_device_ = evaluate_k_space(
						Qabs, x, extract_tmap_gradient )

					log_file.write(
						'(diffraction) Q step = ' + str( Q_idx ) +
						' out of ' + str( num_Q_iterations - 1 ) +
						', Optimization step = ' + str( iteration_counter_diffraction ) +
						 ', and loss = ' + str( total_intensity.item() ) + '\n' )

					print( '(diffraction) Q step = ', Q_idx, ' out of ', ( num_Q_iterations - 1 ), ', Optimization step = ', iteration_counter_diffraction, ', and loss = ', total_intensity.item() )
					iteration_counter_diffraction += 1
					photonic_batch_idx_diffraction = ( photonic_batch_idx_diffraction + 1 ) % number_batches

					elapsed_time = time.time() - start_time
					log_file.write(
						'It took ' + str( elapsed_time ) + ' seconds to run one iteration!\n' )

					log_file.close()

					return total_intensity.item()

			log_file.close()


			if not enable_method_of_moving_asymptotes_diffraction:
				for idx in range( 0, num_photonic_grad_descent_iter ):
					gradn = np.zeros( optimization_variable.shape )

					fom = function_opt_diffraction( optimization_variable, gradn )
					normalize = gradn / np.max( np.abs( gradn ) )

					optimization_variable += gradient_descent_step_size_normalized * normalize
					optimization_variable = np.minimum( device_permittivity, np.maximum( device_background_permittivity, optimization_variable ) )
			
			optimization = nlopt.opt( nlopt.LD_MMA, total_dof )
			optimization.set_lower_bounds( lower_bounds )
			optimization.set_upper_bounds( upper_bounds )

			optimization.set_xtol_rel( optimization_tolerance )
			optimization.set_maxeval( num_photonic_grad_descent_iter )
			optimization.set_max_objective( function_opt_diffraction )

			if enable_method_of_moving_asymptotes_diffraction:
				optimization_variable = optimization.optimize( optimization_variable )




			#
			# todo: regularizer?
			#

			nn_model.train()

			#
			# Run through a couple of rounds of training here - this will get the batch norm going!
			#
			nn_model.train()

			log_file = open( data_folder + "/log.txt", 'a' )

			for nn_epoch in range( 0, num_nn_epochs_per_photonic_epoch ):
				average_loss = 0
				for nn_batch_idx in range( 0, number_batches ):

					batch_start_idx = training_batch_start_idxs[ nn_batch_idx ]
					batch_end_idx = training_batch_end_idxs[ nn_batch_idx ]
					number_in_batch = batch_end_idx - batch_start_idx

					input_tmap = np.pad(
							transmission_map_device_,
							(
								( 0, 0 ), ( 0, 0 ),
								( kx_pad, kx_pad ),
								( ky_pad, ky_pad ),
								( 0, 0 ), ( 0, 0 ) ),
							mode='constant' )
					input_Ek = training_Ek[ batch_start_idx : batch_end_idx ]

					output_I = np.zeros( ( number_in_batch, num_orders, img_dimension, img_dimension ), dtype=np.float32 )
					for batch_idx in range( 0, number_in_batch ):
						for order_idx in range( 0, num_orders ):
							output_E = weight_by_order[ order_idx] * np.fft.ifft2( np.fft.ifftshift( input_Ek[ batch_idx ] * input_tmap[ 0, 0, :, :, 1, order_idx ] ) )
							output_I[ batch_idx, order_idx ] = np.abs( output_E )**2

					output_I = torch.from_numpy( output_I )
					ground_truth_output = torch.squeeze( torch.from_numpy( ground_truth_training[ batch_start_idx : batch_end_idx ] ) )

					nn_optimizer.zero_grad()

					nn_model_output = torch.squeeze( nn_model( output_I ) )


					eval_loss = nn_loss_fn( nn_model_output, ground_truth_output )

					eval_loss.backward()

					nn_optimizer.step()

					average_loss += ( eval_loss.item() / number_batches ) 

					print( eval_loss.item() )

				log_file.write( 'Average loss for epoch ' + str( nn_epoch ) + ' is ' + str( average_loss ) + "\n" )
				# print( 'average loss for epoch ' + str( nn_epoch ) + ' is ' + str( average_loss ) )

			log_file.close()




			def eval_model_by_dataset( dataset_Ek_, ground_truth_ ):
				nn_model.eval()

				input_tmap = np.pad(
						transmission_map_device_,
						(
							( 0, 0 ), ( 0, 0 ),
							( kx_pad, kx_pad ),
							( ky_pad, ky_pad ),
							( 0, 0 ), ( 0, 0 ) ),
						mode='constant' )
				input_Ek = dataset_Ek_

				output_I = np.zeros( ( len( dataset_Ek_ ), num_orders, img_dimension, img_dimension ), dtype=np.float32 )
				for batch_idx in range( 0, len( dataset_Ek_ ) ):
					for order_idx in range( 0, num_orders ):
						output_E = weight_by_order[ order_idx ] * np.fft.ifft2( np.fft.ifftshift( input_Ek[ batch_idx ] * input_tmap[ 0, 0, :, :, 1, order_idx ] ) )
						output_I[ batch_idx, order_idx ] = np.abs( output_E )**2


				output_I = torch.from_numpy( output_I )
				ground_truth_output = torch.squeeze( torch.from_numpy( ground_truth_ ) )

				nn_optimizer.zero_grad()

				nn_model_output = torch.squeeze( nn_model( output_I ) )

				eval_loss = nn_loss_fn( nn_model_output, ground_truth_output )

				return eval_loss.item()

			training_error[ Q_idx ] = eval_model_by_dataset( training_Ek, ground_truth_training )
			validation_error[ Q_idx ] = eval_model_by_dataset( validation_Ek, ground_truth_validation )

			nn_model.eval()


			def function_nlopt( x, gradn ):
				log_file = open( data_folder + "/log.txt", 'a' )

				start_time = time.time()
				global iteration_counter
				global dynamic_weights
				global transmission_map_device_
				global photonic_batch_idx

				# print( 'photonic batch = ' + str( photonic_batch_idx ) )

				# figure_of_merit, gradn[ : ], figure_of_merit_individual_, transmission_map_device_ = evaluate_k_space( Qabs, x )

				batch_start_idx = training_batch_start_idxs[ photonic_batch_idx ]
				batch_end_idx = training_batch_end_idxs[ photonic_batch_idx ]
				number_in_batch = batch_end_idx - batch_start_idx

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
				input_Ek = training_Ek[ batch_start_idx : batch_end_idx ]
				input_Ek = torch.from_numpy( input_Ek )

				output_I = torch.zeros( ( number_in_batch, num_orders, img_dimension, img_dimension ) )
				for batch_idx in range( 0, number_in_batch ):
					for order_idx in range( 0, num_orders ):
						output_E = weight_by_order[ order_idx ] * torch.fft.ifft2( torch.fft.ifftshift( input_Ek[ batch_idx ] * input_tmap[ 0, 0, :, :, 1, order_idx ] ) )
						output_I[ batch_idx, order_idx ] = torch.abs( output_E )**2


				nn_model_output = torch.squeeze( nn_model( output_I ) )
				ground_truth_output = torch.squeeze( torch.from_numpy( ground_truth_training[ batch_start_idx : batch_end_idx ] ) )

				eval_loss = nn_loss_fn( nn_model_output, ground_truth_output )
				eval_loss.backward()

				tmap_gradient = input_tmap.grad.detach().numpy()

				# othergrad = output_I.grad.detach().numpy()

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
					:, : ]

				figure_of_merit_, gradn[ : ], figure_of_merit_individual_, transmission_map_device_ = evaluate_k_space(
					Qabs, x, extract_tmap_gradient )

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
				iteration_counter += 1
				photonic_batch_idx = ( photonic_batch_idx + 1 ) % number_batches

				elapsed_time = time.time() - start_time
				log_file.write(
					'It took ' + str( elapsed_time ) + ' seconds to run one iteration!\n' )

				log_file.close()

				return eval_loss.item()

				# return figure_of_merit


			log_file = open( data_folder + "/log.txt", 'a' )

			
			# last_fom = 0
			# num_photonic_grad_descent_iter = number_batches
			if not enable_method_of_moving_asymptotes:
				for idx in range( 0, num_photonic_grad_descent_iter ):
					gradn = np.zeros( optimization_variable.shape )

					fom = function_nlopt( optimization_variable, gradn )
					normalize = gradn / np.max( np.abs( gradn ) )

					log_file.write(
						'Q step = ' + str( Q_idx ) +
						' out of ' + str( num_Q_iterations - 1 ) +
						', Optimization step = ' + str( idx ) +
						 ', and loss = ' + str( fom ) + '\n' )

					optimization_variable -= gradient_descent_step_size_normalized * normalize
					optimization_variable = np.minimum( device_permittivity, np.maximum( device_background_permittivity, optimization_variable ) )
			
			log_file.close()



			# scipy_result = scipy_optimize.minimize( scipy_fom_fn, optimization_variable, jac=scipy_grad_fn, method='CG', bounds=all_bounds, options=scipy_options )
			# optimization_variable = scipy_result.x

			optimization = nlopt.opt( nlopt.LD_MMA, total_dof )
			optimization.set_lower_bounds( lower_bounds )
			optimization.set_upper_bounds( upper_bounds )

			optimization.set_xtol_rel( optimization_tolerance )
			optimization.set_maxeval( num_photonic_grad_descent_iter )
			optimization.set_min_objective( function_nlopt )

			if enable_method_of_moving_asymptotes:
				optimization_variable = optimization.optimize( optimization_variable )

			# figure_of_merit_, gradn_, figure_of_merit_individual, transmission_map_device = evaluate_k_space( Qabs, optimization_variable )

			# log_file.close()

			np.save( data_folder + '/optimization_variable_' + str( Q_idx ) + '.npy', optimization_variable )
			# np.save( data_folder + '/transmission_map_' + str( Q_idx ) + '.npy', transmission_map_device )

			np.save( data_folder + '/training_error.npy', training_error )
			np.save( data_folder + '/validation_error.npy', validation_error )

			if ( Q_idx == ( num_Q_iterations - 1 ) ):
				np.save( data_folder + '/optimization_variable.npy', optimization_variable )
				np.save( data_folder + '/transmission_map.npy', transmission_map_device_ )
				torch.save( nn_model.state_dict(), data_folder + '/final_model.pt' )


		terminate_message = { "instruction" : "terminate" }
		for processor_rank_idx_ in range( 1, resource_size ):
			comm.send( terminate_message, dest=processor_rank_idx_ )



	elif mode == 'eval':

		optimization_variable = np.load( override_data_folder + '/optimization_variable.npy' )

		optimized_permittivity = np.reshape( optimization_variable, ( num_design_layers * Nx, Ny ) )
		plt.subplot( 1, 2, 1 )
		plt.imshow( optimized_permittivity )
		plt.colorbar()
		plt.subplot( 1, 2, 2 )
		plt.imshow( np.reshape( init_optimization_variable, ( num_design_layers * Nx, Ny ) ) )
		plt.colorbar()
		plt.show() 


		nn_model = torch.nn.Sequential(
			torch.nn.BatchNorm2d( num_orders ),
			torch.nn.Conv2d( num_orders, num_orders // 2, 3, padding=1 ),
			torch.nn.BatchNorm2d( num_orders // 2 ),
			torch.nn.ReLU(),
			torch.nn.Conv2d( num_orders // 2, num_orders // 2, 3, padding=1 ),
			torch.nn.BatchNorm2d( num_orders // 2 ),
			torch.nn.ReLU(),
			torch.nn.Conv2d( num_orders // 2, 1, 3, padding=1 )
		)
		# nn_optimizer = torch.optim.SGD( nn_model.parameters(), lr=learning_rate, momentum=training_momentum )
		nn_loss_fn = torch.nn.MSELoss()

		nn_model.load_state_dict( torch.load( override_data_folder + '/final_model.pt' ) )

		transmission_map_device_ = np.load( override_data_folder + '/transmission_map.npy' )


		# figure_of_merit, gradn, figure_of_merit_individual_, transmission_map_device_ = evaluate_k_space( Qabs, optimization_variable, blank_gradient )



		def eval_model_by_dataset( dataset_Ek_, ground_truth_ ):
			nn_model.eval()

			input_tmap = np.pad(
					transmission_map_device_,
					(
						( 0, 0 ), ( 0, 0 ),
						( kx_pad, kx_pad ),
						( ky_pad, ky_pad ),
						( 0, 0 ), ( 0, 0 ) ),
					mode='constant' )
			input_Ek = dataset_Ek_

			output_I = np.zeros( ( len( dataset_Ek_ ), num_orders, img_dimension, img_dimension ), dtype=np.float32 )
			for batch_idx in range( 0, len( dataset_Ek_ ) ):
				for order_idx in range( 0, num_orders ):
					output_E = weight_by_order[ order_idx ] * np.fft.ifft2( np.fft.ifftshift( input_Ek[ batch_idx ] * input_tmap[ 0, 0, :, :, 1, order_idx ] ) )
					# output_E = np.fft.ifft2( np.fft.ifftshift( input_Ek[ batch_idx ] * input_tmap[ 0, 0, :, :, 1, order_idx ] ) )
					output_I[ batch_idx, order_idx ] = np.abs( output_E )**2


			output_I = torch.from_numpy( output_I )
			ground_truth_output = torch.squeeze( torch.from_numpy( ground_truth_ ) )
			nn_model_output = torch.squeeze( nn_model( output_I ) )

			print('order up')
			for order_idx in range( 0, num_orders ):
				plt.subplot( 2, 3, order_idx + 1 )
				plt.imshow( output_I[ 0, order_idx ] )
				plt.colorbar()
			plt.show()

			plt.subplot( 1, 2, 1 )
			plt.imshow( nn_model_output.detach().numpy()[ 0 ] )
			plt.colorbar()
			plt.subplot( 1, 2, 2 )
			plt.imshow( ground_truth_[ 0 ] )
			plt.colorbar()
			# plt.subplot( 1, 3, 3 )
			# plt.imshow( np.angle( training_samples[ 0 ] ) )
			# plt.colorbar()
			plt.show()
			# plt.imshow( np.abs( nn_model_output.detach().numpy()[ 0 ] - ground_truth_[ 0 ] )**2 )
			# plt.colorbar()
			# plt.show()

			plt.plot( nn_model_output.detach().numpy()[ 0 ][ 51, : ] )
			plt.plot( ground_truth_[ 0 ][ 51, : ], color='g' )
			plt.show()
			plt.plot( nn_model_output.detach().numpy()[ 0 ][ :, 51 ] )
			plt.plot( ground_truth_[ 0 ][ :, 51 ], color='g' )
			plt.show()

			# return eval_loss.item()

		plt.imshow( np.abs( training_samples[ 0 ] ) )
		plt.colorbar()
		plt.show()

		eval_model_by_dataset( training_Ek[ 0 : 3 ], ground_truth_training[ 0 : 3 ] )
		eval_model_by_dataset( validation_Ek[ 0 : 3 ], ground_truth_validation[ 0 : 3 ] )


		adsf



		init_permittivity = np.zeros( Nx * Ny * num_design_layers )
		for layer_idx in range( 0, num_design_layers ):
			blank_density = np.zeros( ( Nx, Ny ) )

			if device_initialization == 'random':
				init_layer_density = np.random.random( blank_density.shape )
				init_layer_density = gaussian_filter( init_layer_density, sigma=initialization_blur_sigma )

				init_layer_density = init_layer_density - np.mean( init_layer_density )# + initialization_mean
				init_layer_density = init_layer_density * initialization_std / np.std( init_layer_density )
				init_layer_density += initialization_mean
				init_layer_density = np.minimum( 1.0, np.maximum( init_layer_density, 0.0 ) )

			elif device_initialization == 'uniform':
				init_layer_density = initiailization_uniform_density * np.ones( blank_density.shape )
			else:
				print( 'Unrecognized device initialization strategy!' )
				sys.exit( 1 )

			init_layer_permittivity = device_background_permittivity + ( device_permittivity - device_background_permittivity ) * init_layer_density

			init_permittivity[ layer_idx * Nx * Ny : ( layer_idx + 1 ) * Nx * Ny ] = init_layer_permittivity.flatten()

		init_optimization_variable[ 0 : device_dof_end ] = init_permittivity


		optimization_variable = init_optimization_variable


		Qabs = np.inf

		fake_gradient = np.ones( desired_transmission.shape, dtype=np.complex )


		if processor_rank == 0:

			device_permittivity_ = preprocess_optimization_variable( optimization_variable )
			figure_of_merit_, gradn_, figure_of_merit_individual, transmission_map_device = evaluate_k_space( Qabs, device_permittivity_, fake_gradient )

			E_input = training_samples[ 0 ]
			Ek_input = np.fft.fftshift( np.fft.fft2( E_input ) )


			kx_start = ( img_dimension // 2 ) - ( num_kx // 2 )
			ky_start = ( img_dimension // 2 ) - ( num_ky // 2 )
			kx_pad = ( img_dimension - num_kx ) // 2
			ky_pad = ( img_dimension - num_ky ) // 2

			for kx_idx in range( 0, num_kx ):
				for ky_idx in range( 0, num_ky ):

					kx = kx_values[ kx_idx ]
					ky = ky_values[ ky_idx ]

					kr = np.sqrt( kx**2 + ky**2 )
					kr_max = ( 2 * np.pi * frequencies[ 0 ] ) * numerical_aperture

					if kr >= kr_max:
						Ek_input[ kx_idx + kx_start, ky_idx + ky_start ] = 0




			# nn_model = torch.nn.Sequential(
			# 	# torch.nn.BatchNorm2d( num_orders ),
			# 	torch.nn.Conv2d( num_orders, 1, 3, padding=1 )
			# 	# num_orders // 2, 3, padding=1 ),
			# 	# torch.nn.ReLU(),
			# 	# torch.nn.Conv2d( num_orders // 2, num_orders // 2, 3, padding=1 ),
			# 	# torch.nn.ReLU(),
			# 	# torch.nn.Conv2d( num_orders // 2, 1, 3, padding=1 )
			# )

			class simple_net(torch.nn.Module):
				def __init__(self):
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

			nn_model = simple_net()

			nn_inputs = np.zeros( ( number_training_samples, num_orders, img_dimension, img_dimension ), dtype=np.float32 )
			gt_samples = np.zeros( ( number_training_samples, img_dimension, img_dimension ), dtype=np.float32 )
			for sample_idx in range( 0, number_training_samples ):
				get_phase = np.angle( training_samples[ sample_idx ] )

				pad_phase_angle = np.pad( np.angle( get_phase ), ( ( 1, 1 ), ( 0, 0 ) ), mode='edge' )
				analytical_phase_grad = ( pad_phase_angle[ 2 :, : ] - pad_phase_angle[ 0 : -2, : ] ) / ( 2 * img_mesh )

				gt_samples[ sample_idx, :, : ] = analytical_phase_grad


				pad_transmission_map_device = np.pad( transmission_map_device, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ), mode='constant' )

				Ek_amplitude_by_order = []
				for order_idx in range( 0, num_orders ):
					#
					# fixing to p-pol for now
					#
					nn_inputs[ sample_idx, order_idx, :, : ] = np.abs( np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device[ 0, 0, :, :, 1, order_idx ] ) ) )**2


			learning_rate = .001
			training_momentum = 0.9

			# nn_optimizer = torch.optim.SGD( nn_model.parameters(), lr=learning_rate, momentum=training_momentum )

			nn_loss_fn = torch.nn.MSELoss()

			nn_model.train()

			nn_model_input = torch.from_numpy( nn_inputs )

			# num_epochs = 0
			# for epoch_idx in range( 0, num_epochs ):
			# 	phase_ground_truths = torch.from_numpy( gt_samples )

			# 	nn_model_output = nn_model( nn_model_input )

			# 	eval_loss = nn_loss_fn( torch.squeeze( nn_model_output ), phase_ground_truths )

			# 	print( 'loss ' + str( eval_loss.item() ) )

			# 	nn_optimizer.zero_grad()

			# 	fom = eval_loss.item()

			# 	eval_loss.backward()

			# 	nn_optimizer.step()



			nn_model.eval()

			Ek_input_gt = torch.from_numpy( gt_samples[ 0 ] )
			Ek_input_gt = Ek_input_gt.unsqueeze( 0 )

			pad_transmission_map_device = np.pad( transmission_map_device, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ), mode='constant' )

			def function_transmision_map_to_reconstruction( transmission_map_device_ ):

				pad_transmission_map_device_ = np.pad( transmission_map_device_, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ), mode='constant' )

				Ek_amplitude_by_order = []
				for order_idx in range( 0, num_orders ):
					#
					# fixing to p-pol for now
					#
					Ek_amplitude_by_order.append( np.abs( np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device_[ 0, 0, :, :, 1, order_idx ] ) ) )**2 )


				Ek_amplitude_by_order = np.array( Ek_amplitude_by_order, dtype=np.float32 )

				nn_model_input = torch.from_numpy( Ek_amplitude_by_order._value )
				nn_model_input = nn_model_input.unsqueeze( 0 )

				nn_model_input.requires_grad = True

				nn_model_output = nn_model( nn_model_input )

				eval_loss = nn_loss_fn( torch.squeeze( nn_model_output ), Ek_input_gt )

				eval_loss.backward()

				get_grad = np.squeeze( nn_model_input.grad.detach().numpy() )


				# figure_of_merit += nn_loss_fn( nn_model_output, Ek_input_gt )

				print( Ek_amplitude_by_order.shape )
				print( get_grad.shape )

				figure_of_merit = 0
				for order_idx in range( 0, num_orders ):
					figure_of_merit += np.sum( Ek_amplitude_by_order[ order_idx ] * get_grad[ order_idx ] )

				return figure_of_merit


			# you have (dL/dE(n))
			# you want dL/dtk
			# dL/dtk = sum( dL/dE(n) * dE(n)/dtk )
			# sum( ( dL/dE(n) * tk )

			# Iout = ifft( Ek * tk ) * conj( ifft( Ek * tk ) ) --> dIout(x) / dtm = conj( ifft( Ek * tk ) ) * Em * e^(imx)
			# dL/tk = sum over x ( dL/dIout(x) * dIout(x)/dtk ) = sum over x ( dL/dIout(x) * Ek * e^(ikx) ) = Ek * conj( E_out ) * ifft( dl/dIout )

			input_tmap = torch.from_numpy( np.pad( transmission_map_device, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ), mode='constant' ) )
			input_tmap.requires_grad = True
			input_Ek = torch.from_numpy( Ek_input )



			# print( transmission_map_device )
			# print( input_Ek.shape )
			# print( Ek_input.shape )
			# print( pad_transmission_map_device.shape )
			# asdf

			# output_E = input_tmap[]

			pad_transmission_map_device = np.pad( transmission_map_device, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ) )

			output_E = np.zeros( ( num_orders, img_dimension, img_dimension ), dtype=np.complex )
			output_I = torch.zeros( ( num_orders, img_dimension, img_dimension ) )
			for order_idx in range( 0, num_orders ):
				output_E[ order_idx ] = weight_by_order[ order_idx ] * np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device[ 0, 0, :, :, 1, order_idx ] ) )
				# output_I[ order_idx ] = np.abs( np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device[ 0, 0, :, :, 1, order_idx ] ) ) )**2

				# needs an ifftshift
				# # output_I[ order_idx ] = torch.abs(  )**2
				alpha = torch.fft.ifft2( torch.fft.ifftshift( input_Ek * input_tmap[ 0, 0, :, :, 1, order_idx ] ) )
				output_I[ order_idx ] = torch.abs( alpha )**2# alpha * torch.conj( alpha )

				# output_I[ order_idx ] = torch.abs( input_Ek * input_tmap[ 0, 0, :, :, 1, order_idx ] )**2

			# plt.plot( np.abs( output_E[ order_idx ][ output_E.shape[ 1 ] // 2 ] )**2 )
			# plt.plot( output_I[ order_idx ][ output_E.shape[ 1 ] // 2 ].detach().numpy(), linestyle='--' )
			# plt.show()
			# asdf

			# output_I = torch.from_numpy( output_I )

			# print( torch.max( output_I ) )
			# print( torch.mean( output_I ) )
			# adsf
			nn_model_input = output_I.unsqueeze( 0 )
			# nn_model_input.requires_grad = True
			nn_model_output = nn_model( nn_model_input )

			eval_loss = nn_loss_fn( torch.squeeze( nn_model_output ), torch.squeeze( Ek_input_gt ) )

			print( 'loss after nn = ' + str( eval_loss.item() ) )

			first_loss = eval_loss.item()

			eval_loss.backward()

			# get_grad = np.squeeze( nn_model_input.grad.detach().numpy() )
			full_grad2 = input_tmap.grad.detach().numpy()

			# print( full_grad2 )

			# print( full_grad2 )
			# print( np.max( np.imag( full_grad2 ) ) )
			# print( np.max( np.real( full_grad2 )))
			# print('----')

			# full_grad = np.zeros( desired_transmission.shape, dtype=np.complex )
			# for order_idx in range( 0, num_orders ):
			# 	full_grad[ 0, 0, :, :, 1, order_idx ] = input_Ek * np.conj( output_E[ order_idx ] ) * np.fft.ifft2( np.fft.ifftshift( get_grad ) )

			# print( full_grad )
			# asdf


			# function_transmision_map_to_reconstruction_grad = grad( function_transmision_map_to_reconstruction )
			# transmision_map_gradient = function_transmision_map_to_reconstruction_grad( transmission_map_device )

			# fom = function_transmision_map_to_reconstruction( transmission_map_device )

			# print( 'fom = ' + str( fom ) )

			#
			# need to extract
			# print( full_grad2.shape )
			# print( transmission_map_device.shape )

			mid_grad2_x = full_grad2.shape[ 2 ] // 2
			mid_grad2_y = full_grad2.shape[ 3 ] // 2

			tmap_size_x = transmission_map_device.shape[ 2 ] 
			tmap_size_y = transmission_map_device.shape[ 3 ]

			tmap_offset_x = tmap_size_x // 2
			tmap_offset_y = tmap_size_y // 2

			extract_grad2 = full_grad2[
				:, :,
				( mid_grad2_x - tmap_offset_x ) : ( mid_grad2_x - tmap_offset_x + tmap_size_x ),
				( mid_grad2_y - tmap_offset_y ) : ( mid_grad2_y - tmap_offset_y + tmap_size_y ),
				:, : ]


			# print(  np.max(np.abs(extract_grad2)))
			# asdf






			test_new_tmap = transmission_map_device + 0.1 * extract_grad2 / np.max( np.abs( extract_grad2 ) )

			# print( extract_grad2[ 0, 0, 0, 0, 0, 0 ] )
			# print( extract_grad2[ 0, 0, 0, 0, 1, 0 ] )
			# asdf


			figure_of_merit_, gradn_, figure_of_merit_individual, transmission_map_device = evaluate_k_space( Qabs, device_permittivity_, extract_grad2 )#test_new_tmap )# extract_grad2 )# test_new_tmap )#extract_grad2 )
			prev_tmap = transmission_map_device.copy()

			# print( extract_grad2[ :, :, :, :, 1, : ] )
			# adsf


			save_grad = gradn_.copy()
			# save_grad = np.random.random( save_grad.shape ) - 0.4
			normalize_grad = save_grad / np.max( np.abs( save_grad ) )



			#
			# Check dL/dtk for the real and imaginary parts of the loss function gradient wrt to the transmission map
			#
			'''
			h = 1e-5#0.001
			num_fd_t = num_orders

			fd_real_t = np.zeros( num_fd_t )
			fd_imag_t = np.zeros( num_fd_t )

			for idx in range( 0, num_fd_t ):
				copy_t_map = transmission_map_device.copy()

				copy_t_map[ 0, 0, 1, 1, 1, idx ] += h


				input_tmap = torch.from_numpy( np.pad( copy_t_map, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ), mode='constant' ) )
				input_tmap.requires_grad = True
				input_Ek = torch.from_numpy( Ek_input )

				# output_E = input_tmap[]

				pad_transmission_map_device = np.pad( copy_t_map, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ) )

				output_E = np.zeros( ( num_orders, img_dimension, img_dimension ), dtype=np.complex )
				output_I = torch.zeros( ( num_orders, img_dimension, img_dimension ) )#, dtype=np.float32 )
				for order_idx in range( 0, num_orders ):
					output_E[ order_idx ] = np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device[ 0, 0, :, :, 1, order_idx ] ) )
					# output_I[ order_idx ] = np.abs( np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device[ 0, 0, :, :, 1, order_idx ] ) ) )**2

					# needs an ifftshift
					output_I[ order_idx ] = torch.abs( torch.fft.ifft2( torch.fft.ifftshift( input_Ek * input_tmap[ 0, 0, :, :, 1, order_idx ] ) ) )**2

				# output_I = torch.from_numpy( output_I )

				nn_model_input = output_I.unsqueeze( 0 )
				# nn_model_input.requires_grad = True
				nn_model_output = nn_model( nn_model_input )

				eval_loss = nn_loss_fn( torch.squeeze( nn_model_output ), torch.squeeze( Ek_input_gt ) )

				second_loss = eval_loss.item()

				fd_real_t[ idx ] = ( second_loss - first_loss ) / h

			for idx in range( 0, num_fd_t ):
				copy_t_map = transmission_map_device.copy()

				copy_t_map[ 0, 0, 1, 1, 1, idx ] += 1j * h


				input_tmap = torch.from_numpy( np.pad( copy_t_map, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ), mode='constant' ) )
				input_tmap.requires_grad = True
				input_Ek = torch.from_numpy( Ek_input )

				# output_E = input_tmap[]

				pad_transmission_map_device = np.pad( copy_t_map, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ) )

				output_E = np.zeros( ( num_orders, img_dimension, img_dimension ), dtype=np.complex )
				output_I = torch.zeros( ( num_orders, img_dimension, img_dimension ) )#, dtype=np.float32 )
				for order_idx in range( 0, num_orders ):
					output_E[ order_idx ] = np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device[ 0, 0, :, :, 1, order_idx ] ) )
					# output_I[ order_idx ] = np.abs( np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device[ 0, 0, :, :, 1, order_idx ] ) ) )**2

					# needs an ifftshift
					output_I[ order_idx ] = torch.abs( torch.fft.ifft2( torch.fft.ifftshift( input_Ek * input_tmap[ 0, 0, :, :, 1, order_idx ] ) ) )**2

				# output_I = torch.from_numpy( output_I )

				nn_model_input = output_I.unsqueeze( 0 )
				# nn_model_input.requires_grad = True
				nn_model_output = nn_model( nn_model_input )

				eval_loss = nn_loss_fn( torch.squeeze( nn_model_output ), torch.squeeze( Ek_input_gt ) )

				second_loss = eval_loss.item()

				fd_imag_t[ idx ] = ( second_loss - first_loss ) / h
				

			print( fd_real_t )
			print( np.real( extract_grad2[ 0, 0, 1, 1, 1, : ] ) )
			print('---')
			print( np.imag( extract_grad2[ 0, 0, 1, 1, 1, : ] ) )
			print( fd_imag_t )

			plt.plot( fd_real_t )
			plt.plot( np.real( extract_grad2[ 0, 0, 1, 1, 1, : ] ), linestyle='--' )
			plt.show()
			plt.plot( fd_imag_t )
			plt.plot( np.imag( extract_grad2[ 0, 0, 1, 1, 1, : ] ), linestyle='--' )
			plt.show()
			# asdf


			num_fd = 11
			h = 0.025#0.05
			fd_size = 2
			fd_offset = fd_size
			fd_up = np.zeros( num_fd )
			fd_down = np.zeros( num_fd )
			adj_grad = np.zeros( num_fd )

			fd_x = Nx // 2
			fd_y_start = 0# Ny // 2



			save_t_map = transmission_map_device.copy()

			for fd_idx in range( 0, num_fd ):
				fd_start = fd_y_start + fd_idx * fd_offset

				device_permittivity_fd = np.reshape( device_permittivity_.copy(), ( Nx, Ny ) )

				# plt.imshow( device_permittivity_fd )
				# plt.colorbar()
				# plt.show()
				# asdf

				device_permittivity_fd[ fd_x : ( fd_x + fd_size ), ( fd_start + fd_idx * fd_offset ) : ( fd_start + fd_idx * fd_offset + 1 * fd_size ) ] += h
				device_permittivity_fd = device_permittivity_fd.flatten()

				print( 'fd idx = ' + str( fd_idx ) )
				# device_permittivity_fd = device_permittivity_.copy()
				# device_permittivity_fd[ fd_start : ( fd_start + fd_size ) ] += 10
				figure_of_merit_, gradn_, figure_of_merit_individual, transmission_map_device = evaluate_k_space( Qabs, device_permittivity_fd, extract_grad2 )# test_new_tmap )


				input_tmap = torch.from_numpy( np.pad( transmission_map_device, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ), mode='constant' ) )
				input_tmap.requires_grad = True
				input_Ek = torch.from_numpy( Ek_input )

				# print( transmission_map_device[ :, :, :, :, 1, 0 ] )
				# print( save_t_map[ :, :, :, :, 1, 0 ] )
				print( np.real( save_t_map[ :, :, :, :, 1, 0 ] - transmission_map_device[ :, :, :, :, 1, 0 ] ) )
				print('----')
				print( np.imag( save_t_map[ :, :, :, :, 1, 0 ] - transmission_map_device[ :, :, :, :, 1, 0 ] ) )
				# asdf

# [[[[-1.67601709e-05 -1.51445716e-05 -8.57880953e-06]
#    [-2.30311652e-05  1.63283043e-05 -1.96946275e-05]
#    [-7.23033809e-06 -1.57755822e-05 -1.73006196e-05]]]]
# ----
# [[[[-7.90639183e-06 -1.50357440e-05 -8.85363737e-06]
#    [-8.07822568e-06 -3.20954430e-05 -1.28220869e-05]
#    [-1.28756037e-05 -1.39261535e-05 -4.66924134e-06]]]]

				# output_E = input_tmap[]

				pad_transmission_map_device = np.pad( transmission_map_device, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ) )

				output_E = np.zeros( ( num_orders, img_dimension, img_dimension ), dtype=np.complex )
				output_I = torch.zeros( ( num_orders, img_dimension, img_dimension ) )#, dtype=np.float32 )
				for order_idx in range( 0, num_orders ):
					output_E[ order_idx ] = np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device[ 0, 0, :, :, 1, order_idx ] ) )
					# output_I[ order_idx ] = np.abs( np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device[ 0, 0, :, :, 1, order_idx ] ) ) )**2

					# needs an ifftshift
					output_I[ order_idx ] = torch.abs( torch.fft.ifft2( torch.fft.ifftshift( input_Ek * input_tmap[ 0, 0, :, :, 1, order_idx ] ) ) )**2

				# output_I = torch.from_numpy( output_I )

				nn_model_input = output_I.unsqueeze( 0 )
				# nn_model_input.requires_grad = True
				nn_model_output = nn_model( nn_model_input )

				eval_loss = nn_loss_fn( torch.squeeze( nn_model_output ), torch.squeeze( Ek_input_gt ) )

				second_loss = eval_loss.item()

				print( first_loss )
				print( second_loss )
				print()

				reshape_grad = np.reshape( save_grad.copy(), ( Nx, Ny ) )
				adj_grad[ fd_idx ] = np.sum( reshape_grad[ fd_x : ( fd_x + fd_size ), ( fd_start + fd_idx * fd_offset ) : ( fd_start + fd_idx * fd_offset + 1 * fd_size ) ] )


				# adj_grad[ fd_idx ] = np.sum( save_grad[ fd_start : ( fd_start + fd_size ) ] )

				fd_up[ fd_idx ] = second_loss
				# fd_up[ fd_idx ] = figure_of_merit_


			for fd_idx in range( 0, num_fd ):
				fd_start = fd_y_start + fd_idx * fd_offset

				device_permittivity_fd = np.reshape( device_permittivity_.copy(), ( Nx, Ny ) )

				device_permittivity_fd[ fd_x : ( fd_x + fd_size ), ( fd_start + fd_idx * fd_offset ) : ( fd_start + fd_idx * fd_offset + 1 * fd_size ) ] += (-h)
				device_permittivity_fd = device_permittivity_fd.flatten()

				print( 'fd idx = ' + str( fd_idx ) )
				# device_permittivity_fd = device_permittivity_.copy()
				# device_permittivity_fd[ fd_start : ( fd_start + fd_size ) ] += 10
				figure_of_merit_, gradn_, figure_of_merit_individual, transmission_map_device = evaluate_k_space( Qabs, device_permittivity_fd, extract_grad2 )# test_new_tmap )


				input_tmap = torch.from_numpy( np.pad( transmission_map_device, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ), mode='constant' ) )
				input_tmap.requires_grad = True
				input_Ek = torch.from_numpy( Ek_input )

				# output_E = input_tmap[]

				pad_transmission_map_device = np.pad( transmission_map_device, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ) )

				output_E = np.zeros( ( num_orders, img_dimension, img_dimension ), dtype=np.complex )
				output_I = torch.zeros( ( num_orders, img_dimension, img_dimension ) )#, dtype=np.float32 )
				for order_idx in range( 0, num_orders ):
					output_E[ order_idx ] = np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device[ 0, 0, :, :, 1, order_idx ] ) )
					# output_I[ order_idx ] = np.abs( np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device[ 0, 0, :, :, 1, order_idx ] ) ) )**2

					# needs an ifftshift
					output_I[ order_idx ] = torch.abs( torch.fft.ifft2( torch.fft.ifftshift( input_Ek * input_tmap[ 0, 0, :, :, 1, order_idx ] ) ) )**2

				# output_I = torch.from_numpy( output_I )

				nn_model_input = output_I.unsqueeze( 0 )
				# nn_model_input.requires_grad = True
				nn_model_output = nn_model( nn_model_input )

				eval_loss = nn_loss_fn( torch.squeeze( nn_model_output ), torch.squeeze( Ek_input_gt ) )

				second_loss = eval_loss.item()

				print( first_loss )
				print( second_loss )
				print()

				reshape_grad = np.reshape( save_grad.copy(), ( Nx, Ny ) )
				# adj_grad[ fd_idx ] = np.sum( reshape_grad[ fd_x : ( fd_x + fd_size ), ( fd_start + fd_idx * fd_size ) : ( fd_start + ( fd_idx + 1 ) * fd_size ) ] )


				# adj_grad[ fd_idx ] = np.sum( save_grad[ fd_start : ( fd_start + fd_size ) ] )

				fd_down[ fd_idx ] = second_loss
				# fd_down[ fd_idx ] = figure_of_merit_



			fd = ( np.array( fd_up ) - np.array( fd_down ) ) / ( 2 * h )


			# plt.imshow( np.reshape( save_grad, ( Nx, Ny ) ) )
			# plt.colorbar()
			# plt.show()

			print( fd )
			print( adj_grad )

			fig, ax = plt.subplots()
			# ax2 = ax.twinx()
			ax.plot( fd / np.max( np.abs( fd ) ), color='b' )
			ax.plot( adj_grad / np.max( np.abs( adj_grad ) ), color='g', linestyle='--' )
			plt.show()

			# asdf


			'''




			# print( gradn_ )

			step_sizes = np.linspace( 0, 0.1, 7 )
			changes = []
			expected_changes = []
			device_permittivity_start = device_permittivity_.copy()
			for step_size_idx in range( 0, len( step_sizes ) ):
				step_size = step_sizes[ step_size_idx ]
			# step_size = 0.05

				device_permittivity_ = device_permittivity_start - normalize_grad * step_size

				figure_of_merit_, gradn_, figure_of_merit_individual, transmission_map_device = evaluate_k_space( Qabs, device_permittivity_, extract_grad2 )# test_new_tmap )


				input_tmap = torch.from_numpy( np.pad( transmission_map_device, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ), mode='constant' ) )
				input_tmap.requires_grad = True
				input_Ek = torch.from_numpy( Ek_input )

				# output_E = input_tmap[]

				pad_transmission_map_device = np.pad( transmission_map_device, ( ( 0, 0 ), ( 0, 0 ), ( kx_pad, kx_pad ), ( ky_pad, ky_pad ), ( 0, 0 ), ( 0, 0 ) ) )

				output_E = np.zeros( ( num_orders, img_dimension, img_dimension ), dtype=np.complex )
				output_I = torch.zeros( ( num_orders, img_dimension, img_dimension ) )#, dtype=np.float32 )
				for order_idx in range( 0, num_orders ):
					output_E[ order_idx ] = weight_by_order[ order_idx ] * np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device[ 0, 0, :, :, 1, order_idx ] ) )
					# output_I[ order_idx ] = np.abs( np.fft.ifft2( np.fft.ifftshift( Ek_input * pad_transmission_map_device[ 0, 0, :, :, 1, order_idx ] ) ) )**2

					# needs an ifftshift
					output_I[ order_idx ] = torch.abs( torch.fft.ifft2( torch.fft.ifftshift( input_Ek * input_tmap[ 0, 0, :, :, 1, order_idx ] ) ) )**2

				# output_I = torch.from_numpy( output_I )

				nn_model_input = output_I.unsqueeze( 0 )
				# nn_model_input.requires_grad = True
				nn_model_output = nn_model( nn_model_input )

				eval_loss = nn_loss_fn( torch.squeeze( nn_model_output ), torch.squeeze( Ek_input_gt ) )

				second_loss = eval_loss.item()

				print( 'loss after nn again = ' + str( eval_loss.item() ) )

				expected_change = -np.sum( normalize_grad * step_size * save_grad )
				actual_change = ( second_loss - first_loss )

				changes.append( actual_change )
				expected_changes.append( expected_change )

				print( 'expected change: ' + str( expected_change ) + ' and actual change: ' + str( actual_change ) )
				print()

			plt.plot( step_sizes, changes )
			plt.plot( step_sizes, expected_changes, linestyle='--' )
			plt.show()

			eval_loss.backward()

			# get_grad = np.squeeze( nn_model_input.grad.detach().numpy() )
			full_grad2 = input_tmap.grad.detach().numpy()

			# print( full_grad2 )

			# print( full_grad2 )
			# print( np.max( np.abs( full_grad2 ) ) )


			# fom2 = function_transmision_map_to_reconstruction( transmission_map_device )

			# print( 'fom2 = ' + str( fom2 ) )

			terminate_message = { "instruction" : "terminate" }
			for processor_rank_idx_ in range( 1, resource_size ):
				comm.send( terminate_message, dest=processor_rank_idx_ )



#* If not the master, then this is one of the workers!

else:

	#
	# Run a loop waiting for something to do or the notification that the optimization is done
	#
	while True:

		#
		# This is a blocking receive call, expecting information to be sent from the master
		#
		data = comm.recv( source=0 )

		if data[ "instruction" ] == "terminate":
			break
		elif data[ "instruction" ] == "simulate":
			processor_simulate( data )
		else:
			print( 'Unknown call to master ' + str( processor_rank ) + ' of ' + data[ 'instruction' ] + '\nExiting...' )
			sys.exit( 1 )




