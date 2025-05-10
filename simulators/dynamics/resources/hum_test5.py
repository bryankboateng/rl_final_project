import pybullet as p
import pybullet_data
import time
import numpy as np
from math import pi
from humanoid import Humanoid  # your existing class
from IK_humanoid import IKWalkingController  # assumes you saved controller in this file

# === Initialize PyBullet GUI ===
client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
time.sleep(1)  # let GUI load fully

# === Instantiate and Reset Humanoid ===
humanoid = Humanoid(client)

# === Improved Camera View ===
p.resetDebugVisualizerCamera(
    cameraDistance=10,
    cameraYaw=45,
    cameraPitch=-25,
    cameraTargetPosition=[0, 0, 1.0]
)

# === Instantiate Walking Controller ===
controller = IKWalkingController(humanoid)

# === Run Simulation Loop ===
print("[INFO] Starting walk-in-place simulation.")
dt = 1.0 / 60.0
while True:
    controller.step(dt=dt)  # Apply IK-based walk step
    p.stepSimulation()
    time.sleep(1.0 / 240.0)  # Match PyBullet default timestep
