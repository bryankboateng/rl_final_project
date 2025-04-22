import pybullet as p
import pybullet_data
import time

# Connect to GUI
physicsClient = p.connect(p.GUI)

# Set visualizer options for clean and realistic view
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)        # Hide side panel
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)    # Enable shadows
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # Temporarily disable rendering

# Set path and environment
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For plane.urdf
p.setGravity(0, 0, -9.81)
p.resetSimulation()

# Load plane
planeId = p.loadURDF("plane.urdf")

# Load humanoid URDF
robot_path = "/Users/bboat/Desktop/rl_final_project/simulators/dynamics/resources/humanoid/humanoid.urdf"
start_pos = [0, 0, 1]                  # Slightly above ground
robot_id = p.loadURDF(robot_path, basePosition=start_pos, useFixedBase=False)

# Re-enable rendering now that loading is done
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

# Simulation loop
while True:
    p.stepSimulation()
    time.sleep(1. / 240.)
