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
start_pos = [0, 0, 4]
start_ori = p.getQuaternionFromEuler([1.57, 0, 0])  # <- adjust this until robot stands

robot_id = p.loadURDF(robot_path, basePosition=start_pos, baseOrientation=start_ori, useFixedBase=False)

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

joint_name_to_idx = {}

for joint_idx in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, joint_idx)
    joint_name = info[1].decode("utf-8")
    joint_type = JOINT_TYPES.get(info[2], "UNKNOWN")
    joint_name_to_idx[joint_name] = joint_idx
    print(f"{joint_idx:2d} → {joint_name:20s} → {joint_type}")

# Apply a "standing" pose: slightly bent knees, elbows bent
knee_bend_angle = -0.2  # slight knee bend (for stability)
elbow_bend_angle = 1.5  # elbows bent

# Define desired standing pose
standing_pose = {
    "right_knee": knee_bend_angle,
    "left_knee": knee_bend_angle,
    "right_elbow": elbow_bend_angle,
    "left_elbow": elbow_bend_angle
}

# Apply standing pose
for joint_name, angle in standing_pose.items():
    idx = joint_name_to_idx[joint_name]
    p.resetJointState(robot_id, idx, targetValue=angle)

# Re-enable rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
#(.41, 3.52) (fall down, upright)
# Main simulation loop
while p.isConnected():
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    print(pos)
    p.stepSimulation()
    time.sleep(1. / 240.)
