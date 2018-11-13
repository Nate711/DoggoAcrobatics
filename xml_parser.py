# Read in the file
import os
from os.path import expanduser


###### ROBOT PARAMETERS #####

leg_radius = 0.005
doggo_timestep = 0.001
doggo_solref = doggo_timestep*2


###### GYM PARAMETERS #####

dir_path = os.path.dirname(os.path.realpath(__file__))
in_file = dir_path+"/half_cheetah_variable.xml"

home = expanduser("~")
out_file = home+'/Documents/gym/gym/envs/mujoco/assets/half_cheetah.xml'



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







# Write the file out again
with open(out_file, 'w') as file:
  file.write(filedata)