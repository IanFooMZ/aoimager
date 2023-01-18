import autograd.numpy as np

class DeviceGeometry():

	def __init__( self,
		input_layer_thickness, output_layer_thickness, input_index, output_index,
		device_layer_dimensions, design_layer_thicknesses,
		has_spacers, spacer_layer_thicknesses, spacer_layer_indices, encapsulation_thickness,
		encapsulation_index):

		self.input_layer_thickness = input_layer_thickness
		self.output_layer_thickness = output_layer_thickness
		self.input_index = input_index
		self.output_index = output_index
		self.device_layer_dimensions = device_layer_dimensions
		self.design_layer_thicknesses = design_layer_thicknesses
		self.num_design_layers = len( self.design_layer_thicknesses )
		self.has_spacers = has_spacers
		self.spacer_layer_thicknesses = spacer_layer_thicknesses
		self.spacer_layer_indices = spacer_layer_indices
		self.encapsulation_thickness = encapsulation_thickness
		self.encapsulation_index = encapsulation_index

		if self.has_spacers:
			assert self.num_design_layers == ( len( self.spacer_layer_thicknesses ) + 1 ), (
				"The spacer layer thicknesses should be one less than the number of design layers!" )

		# Where to record the field amplitudes
		self.input_fourier_sampling_offset = self.input_layer_thickness
		self.output_fourier_sampling_offset = 0

		self.simulate_substrate = False

	def add_substrate( self, substrate_thickness, substrate_index ):
		self.substrate_thickness = substrate_thickness
		self.substrate_index = substrate_index

		self.simulate_substrate = True

	def add_layers( self, simulation ):
		'''Called when setting up gRCWA environment to add incident, patterned, and substrate layers with epsilon data.
  		according to the specifications of the DeviceGeometry object.
    	Input: simulation is a gRCWA object	https://grcwa.readthedocs.io/en/latest/usage.html
     	Output: None. simulation is modified accordingly.'''
		# preprocess device, add epsilon data
		# Assume for now that we have square unit cells

		Nx = self.device_layer_dimensions[ 0 ]
		Ny = self.device_layer_dimensions[ 1 ]

		simulation.Add_LayerUniform( self.input_layer_thickness, self.input_index ** 2 )

		if self.simulate_substrate:
			simulation.Add_LayerUniform( self.substrate_thickness, self.substrate_index**2 )

		for layer_idx in range( 0, self.num_design_layers ):

			simulation.Add_LayerGrid( self.design_layer_thicknesses[ layer_idx ], Nx, Ny )

			if self.has_spacers and ( layer_idx < ( self.num_design_layers - 1 ) ):
				simulation.Add_LayerUniform( self.spacer_layer_thicknesses[ layer_idx ], self.spacer_layer_indices[ layer_idx ]**2 )

		# Add encapsulating layer
		simulation.Add_LayerUniform( self.encapsulation_thickness, self.encapsulation_index ** 2 )

		# Add output layer (air)
		simulation.Add_LayerUniform( self.output_layer_thickness, self.output_index ** 2 )
