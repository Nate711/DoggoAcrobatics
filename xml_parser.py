# Read in the file
import os
import shutil
from os.path import expanduser


###### ROBOT PARAMETERS #####

leg_radius = 0.005
doggo_friction = 1.5
doggo_timestep = 0.001
doggo_solref = doggo_timestep*2
doggo_radial_armature = 0.1
doggo_solimp1 = 0.999
doggo_solimp2 = 0.999


###### GYM PARAMETERS #####

dir_path = os.path.dirname(os.path.realpath(__file__))
in_file = dir_path+"/half_cheetah_variable.xml"

home = expanduser("~")
envs_path = '/home/benja/anaconda3/envs/spinningup/lib/python3.6/site-packages/gym/envs/'
mujoco_path = envs_path + 'mujoco/'
out_file = mujoco_path + 'assets/half_cheetah.xml'


### Parse the xml

print('Input xml: %s'%in_file)
print('Output xml: %s'%out_file)

with open(in_file, 'r') as file :
  filedata = file.read()


#### Replace variable names with values ####
# Replace the target string
filedata = filedata.replace('leg_radius', str(leg_radius))
filedata = filedata.replace('doggo_timestep', str(doggo_timestep))
filedata = filedata.replace('doggo_solref', str(doggo_solref))
filedata = filedata.replace('doggo_friction', str(doggo_friction))
filedata = filedata.replace('doggo_radial_armature', str(doggo_radial_armature))
filedata = filedata.replace('doggo_solimp1', str(doggo_solimp1))
filedata = filedata.replace('doggo_solimp2', str(doggo_solimp2))



# Write the file out again
with open(out_file, 'w') as file:
  file.write(filedata)

shutil.copyfile('half_cheetah.py', mujoco_path + 'half_cheetah.py')
shutil.copyfile('__init__.py', envs_path + '__init__.py')