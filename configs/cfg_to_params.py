import numpy as np
import os
import pickle
import shutil
import sys
import yaml
import torch

#* Establish folder structure & location
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
python_src_directory = os.path.dirname(python_src_directory)		# Go up one folder


#* Input Arguments (passed in bash script)
if len( sys.argv ) < 3:							# Print instructions on what input arguments are needed.
	print( "Usage: python " + str( sys.argv[ 0 ] ) + " { yaml file name }" )
	sys.exit( 1 )
 
yaml_filename = sys.argv[ 1 ]

#* Load YAML file
with open( yaml_filename, 'rb' ) as yaml_file:
	parameters = yaml.safe_load( yaml_file )


#* Create project folder to store all files
projects_directory_location = 'ares' + os.path.basename(python_src_directory)
# projects_directory_location = sys.argv[ 2 ]
if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

#* Output / Save Paths
parameters['DATA_FOLDER'] = projects_directory_location
parameters['MODEL_PATH'] = os.path.join(projects_directory_location, 'model.pth')
parameters['OPTIMIZER_PATH'] = os.path.join(projects_directory_location, 'optimizer.pth')


#* Save out the various files that exist right before the optimization runs for debugging purposes.
# If these files have changed significantly, the optimization should be re-run to compare to anything new.

if not os.path.isdir( projects_directory_location + "/saved_scripts/" ):
	os.mkdir( projects_directory_location + "/saved_scripts/" )

shutil.copy2( python_src_directory + "/main.py", projects_directory_location + "/saved_scripts/main.py" )
shutil.copy2( os.path.join(python_src_directory, yaml_filename), 
             projects_directory_location + "/saved_scripts/" + os.path.basename(yaml_filename) )
shutil.copy2( python_src_directory + "/configs/cfg_to_params.py", projects_directory_location + "/saved_scripts/cfg_to_params.py" )
#TODO: et cetera... might have to save out various scripts from each folder



#* Derived Parameters

#TODO: Move from main.py to here


#
#* Debug Options
#

running_on_local_machine = False
slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
if slurm_job_env_variable is None:
    running_on_local_machine = True
    
if running_on_local_machine:	
    lumapi_filepath =  r"C:\Program Files\Lumerical\v212\api\python\lumapi.py"
else:	
    #! do not include spaces in filepaths passed to linux
    lumapi_filepath = r"/central/home/ifoo/lumerical/2021a_r22/api/python/lumapi.py"


start_from_step = 0 	    # 0 if running entire file in one shot
shelf_fn = 'save_state'


print('Config imported and processed.')