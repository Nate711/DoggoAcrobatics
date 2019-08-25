import os
import shutil

# Verify that ENVS_PATH is indeed pointing at your gym installation
home = os.path.expanduser("~")
ENVS_PATH = os.path.join(str(home), "Documents/gym/gym/envs/")
MUJOCO_PATH = os.path.join(ENVS_PATH, "mujoco/")
OUT_FILE = os.path.join(MUJOCO_PATH, "assets/half_cheetah.xml")

shutil.copyfile("half_cheetah.py", MUJOCO_PATH + "half_cheetah.py")
print("Copied half_cheetah.py to Gym directory")
shutil.copyfile("__init__.py", ENVS_PATH + "__init__.py")
print("Copied __init__.py to Gym directory")
shutil.copyfile("pupper_out.xml", OUT_FILE)
print("Copied pupper_out.xml to Gym directory")
