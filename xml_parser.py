# Read in the file
import os
from os.path import expanduser

dir_path = os.path.dirname(os.path.realpath(__file__))
in_file = dir_path+"/half_cheetah_variable.xml"

home = expanduser("~")
out_file = home+'/Documents/gym/gym/envs/mujoco/assets/half_cheetah.xml'

print(in_file)
print(out_file)

with open(in_file, 'r') as file :
  filedata = file.read()


#### Replace variable names with values ####
# Replace the target string
filedata = filedata.replace('leg_radius', '0.005')




# Write the file out again
with open(out_file, 'w') as file:
  file.write(filedata)