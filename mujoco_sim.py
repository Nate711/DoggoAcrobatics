import mujoco_py
model = mujoco_py.load_model_from_path("/Users/nathan/Documents/gym/gym/envs/mujoco/assets/half_cheetah.xml")
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
for i in range(5000):
    viewer.render()
