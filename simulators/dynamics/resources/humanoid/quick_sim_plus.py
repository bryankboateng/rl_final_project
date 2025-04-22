import pybullet as p
import pybullet_data
import time

# Connect to GUI
physicsClient = p.connect(p.GUI)

# Configure visualizer for clarity
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# Load plane and humanoid
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")
robot_path = "/Users/bboat/Desktop/rl_final_project/simulators/dynamics/resources/humanoid/humanoid.urdf"
start_pos = [0, 0, 1]
robot_id = p.loadURDF(robot_path, basePosition=start_pos, useFixedBase=False)

# Optional: Draw frames at link origins
for link_idx in range(-1, p.getNumJoints(robot_id)):
    pos, orn = p.getLinkState(robot_id, link_idx)[:2] if link_idx != -1 else p.getBasePositionAndOrientation(robot_id)
    p.addUserDebugText(
        "base" if link_idx == -1 else f"link_{link_idx}",
        pos,
        textColorRGB=[1, 0, 0],
        textSize=1.2
    )
    p.addUserDebugLine(pos, [pos[0]+0.1, pos[1], pos[2]], [1, 0, 0], 2)  # X-axis
    p.addUserDebugLine(pos, [pos[0], pos[1]+0.1, pos[2]], [0, 1, 0], 2)  # Y-axis
    p.addUserDebugLine(pos, [pos[0], pos[1], pos[2]+0.1], [0, 0, 1], 2)  # Z-axis

# Print joint info with type labels
print("\n[Joint Index → Name → Type]\n")
JOINT_TYPES = {
    p.JOINT_REVOLUTE: "REVOLUTE",
    p.JOINT_PRISMATIC: "PRISMATIC",
    p.JOINT_SPHERICAL: "SPHERICAL",
    p.JOINT_PLANAR: "PLANAR",
    p.JOINT_FIXED: "FIXED",
    p.JOINT_POINT2POINT: "P2P",
}

for joint_idx in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, joint_idx)
    joint_name = info[1].decode("utf-8")
    joint_type = JOINT_TYPES.get(info[2], "UNKNOWN")
    print(f"{joint_idx:2d} → {joint_name:20s} → {joint_type}")

# Re-enable rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

# Run simulation loop
while p.isConnected():
    p.stepSimulation()
    time.sleep(1. / 240.)
