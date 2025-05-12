import pybullet as p
import pybullet_data
import time
import numpy as np
from math import pi
from humanoid import Humanoid  

"""
Self-Written Humanoid Testing Pybullet Code 
Via pybullet api datasheet: https://raw.githubusercontent.com/bulletphysics/bullet3/master/docs/pybullet_quickstartguide.pdf
"""

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
time.sleep(1)  # let GUI load fully first to stop weird crash


humanoid = Humanoid(client)
humanoid.reset([0, 0, 1.5])


p.resetDebugVisualizerCamera(
    cameraDistance=10,
    cameraYaw=45,
    cameraPitch=-25,
    cameraTargetPosition=[0, 0, 1.5]
)


controllable_joints = []
joint_sliders = {}

for idx in humanoid.joint_index:
    joint_info = p.getJointInfo(humanoid.id, idx, physicsClientId=client)
    joint_name = joint_info[1].decode()
    joint_type = joint_info[2]
    lo, hi = joint_info[8], joint_info[9]

    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        controllable_joints.append(idx)
        slider_range = (lo, hi) if hi > lo else (-pi, pi)
        slider_id = p.addUserDebugParameter(joint_name, *slider_range, 0.0)
        joint_sliders[(joint_name, idx)] = slider_id
    else:
        print(f"[Info] Skipping non-controllable joint: {joint_name}")


p.resetBasePositionAndOrientation(
    humanoid.id, [0, 0, 4], p.getQuaternionFromEuler([1.57, 0, 0])
)
for _ in range(240):
    p.stepSimulation()
    time.sleep(1. / 240.)


while p.isConnected():
    target_angles = []

    for idx in controllable_joints:
        joint_name = p.getJointInfo(humanoid.id, idx)[1].decode()
        slider_id = joint_sliders.get((joint_name, idx))

        if slider_id is not None:
            try:
                angle = p.readUserDebugParameter(slider_id, client)
            except p.error:
                print(f"[Warning] Failed to read slider for {joint_name}, defaulting to 0.0")
                angle = 0.0
            target_angles.append(angle)

    humanoid.apply_position(target_angles)
    print("humanoid foot",humanoid.get_foot_contact())
    p.stepSimulation()
    time.sleep(1. / 240.)
