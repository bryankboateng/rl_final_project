import pybullet as p
import pybullet_data
import time
import numpy as np

# === Choose Reset Type: "stand" or "drop" ===
reset_type = "stand"  # Change to "drop" to test drop reset

# Connect to GUI
client=p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)

# Configure visualizer
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# Load plane and humanoid
p.loadURDF("plane.urdf")
robot_path = "/Users/bboat/Desktop/rl_final_project/simulators/dynamics/resources/humanoid/humanoid.urdf"
robot_id = p.loadURDF(robot_path, basePosition=[0, 0, 1], baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), useFixedBase=False)

# Map joint names
joint_name_to_idx = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    joint_name_to_idx[info[1].decode()] = j

# Define target standing pose
standing_pose = {
    "right_knee": 0.0,
    "left_knee": 0.0,
    "right_elbow": 1.5,
    "left_elbow": 1.5
}

# === Reset according to type ===
if reset_type == "drop":
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 4], p.getQuaternionFromEuler([1.57, 0, 0]))
    for _ in range(240):  # simulate falling
        p.stepSimulation()
        time.sleep(1. / 240.)

elif reset_type == "stand":
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 4.0], p.getQuaternionFromEuler([1.57, 0, 0]))

    # Get current pose
    current_pose = {
        name: p.getJointState(robot_id, joint_name_to_idx[name])[0]
        for name in standing_pose
    }

    # Interpolate to standing pose
    N = 100
    trajectory = {
        name: np.linspace(current_pose[name], standing_pose[name], N)
        for name in standing_pose
    }

    for i in range(N):
        for joint_name, angle_seq in trajectory.items():
            p.setJointMotorControl2(
                robot_id, joint_name_to_idx[joint_name],
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle_seq[i],
                force=100
            )
        p.stepSimulation()
        time.sleep(1. / 240.)

# Re-enable rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

# === Main Simulation Loop ===
while p.isConnected():
    pos, orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=client)
    print(pos)
    p.stepSimulation()
    time.sleep(1. / 240.)
