import pybullet as p
import pybullet_data
import time
import numpy as np
from math import pi

# Load your existing Humanoid class from file
from humanoid import Humanoid

# Connect to PyBullet and initialize simulation
client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8, physicsClientId=client)

# Instantiate the Humanoid
humanoid = Humanoid(client)
humanoid.reset([0, 0, 1.5])

# Set an improved camera view
p.resetDebugVisualizerCamera(
    cameraDistance=10,
    cameraYaw=45,
    cameraPitch=-20,
    cameraTargetPosition=[0, 0, 1]
)

# Identify controllable joints (i.e., revolute joints)
controllable_joints = []
joint_sliders = {}

for idx in humanoid.joint_index:
    joint_info = p.getJointInfo(humanoid.id, idx)
    joint_name = joint_info[1].decode()
    joint_type = joint_info[2]
    lo, hi = joint_info[8], joint_info[9]

    if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
        controllable_joints.append(idx)
        if hi > lo:
            slider = p.addUserDebugParameter(joint_name, lo, hi, 0.0)
        else:
            slider = p.addUserDebugParameter(joint_name, -pi, pi, 0.0)
        joint_sliders[joint_name] = slider
    else:
        print(f"[Info] Skipping joint {joint_name} (type {joint_type})")

# Simulate drop and wait
p.resetBasePositionAndOrientation(
    humanoid.id, [0, 0, 4], p.getQuaternionFromEuler([1.57, 0, 0])
)
for _ in range(240):
    p.stepSimulation()
    time.sleep(1. / 240.)

# Control loop
while p.isConnected():
    target_angles = []
    for idx in controllable_joints:
        joint_name = p.getJointInfo(humanoid.id, idx)[1].decode()
        slider = joint_sliders.get(joint_name)
        if slider is not None:
            try:
                angle = p.readUserDebugParameter(slider)
            except p.error:
                print(f"[Warning] Failed to read slider for {joint_name}, defaulting to 0.0")
                angle = 0.0
            target_angles.append(angle)
        else:
            target_angles.append(0.0)

    humanoid.apply_position(target_angles)
    p.stepSimulation()
    time.sleep(1. / 240.)
