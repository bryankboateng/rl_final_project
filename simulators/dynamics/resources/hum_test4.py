import pybullet as p
import pybullet_data
import time
import numpy as np
from math import pi
from humanoid import Humanoid  # your existing class

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
    cameraTargetPosition=[0, 0, 1.5]
)

# === Setup sliders for controllable joints ===
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

# === Optional: Drop pose to see stability ===
p.resetBasePositionAndOrientation(
    humanoid.id, [0, 0, 4], p.getQuaternionFromEuler([1.57, 0, 0])
)
for _ in range(240):
    p.stepSimulation()
    time.sleep(1. / 240.)

# === Control loop ===
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

# Randomize spherical joint targets each frame
    # hip_ankle_targets = [
    #     tuple(np.random.uniform(-0.2, 0.2, size=3)),  # right_hip
    #     tuple(np.random.uniform(-0.2, 0.2, size=3)),  # right_ankle
    #     tuple(np.random.uniform(-0.2, 0.2, size=3)),  # left_hip
    #     tuple(np.random.uniform(-0.2, 0.2, size=3)),  # left_ankle
    # ]
    
    hip_ankle_targets = [
        tuple(np.random.uniform(-np.pi/2, np.pi/2, size=3)),  # right_hip
        tuple(np.zeros(3)),  # right_ankle
        tuple(np.zeros(3)),  # left_hip
        tuple(np.zeros(3)),  # left_ankle
    ]
    
    # === Action Center ===
# [rev] knees: slightly flexed (safer under load), elbows: neutral (~1.5 rad)
# [sph] hips/ankles: upright support pose, slightly inward rotation
    action_center_1 = np.array([
        # Revolute joints (right_knee, left_knee, right_elbow, left_elbow)
        -np.pi/12,  # right_knee   #12
        -np.pi/12,  # left_knee
        np.pi / 2,  # right_elbow
        np.pi / 2,  # left_elbow

    ])
   # apparently the spherical has the following according to visul format # roll, yaw, pitch
    
    action_center_2 = np.array([tuple([0.0, 0.0, np.pi/6]), #bend hip 6 right_hip
                                tuple([0.0, 0.0, 0.0]), # right_ankle
                                tuple([0.0, 0.0, np.pi/6]),  # left_hip
                                tuple([0.0, 0.0, 0.0]), # left_ankle
                                tuple([0.0, 0.0, -np.pi/6])]) # lean forward -12 or -6 chest
    

    
    action_center_3 = np.zeros(4)
    action_center_4 = np.array([tuple(3*[0.0])for _ in range(5)])


    # humanoid.apply_position2(target_angles, hip_ankle_targets)
    print("applying action center")
    humanoid.apply_position2(action_center_3, action_center_2)

    
    # # i want the nonpositive targets dict
    # targets = {k: v for k, v in targets.items() if v <= 0}
    # print("target", targets)
    # pso, vel = humanoid.get_joint_position()
    # angles_chest = p.getEulerFromQuaternion(pso[20:24])
    # angles_left_hip = p.getEulerFromQuaternion(pso[9:13])
    # angles_right_hip = p.getEulerFromQuaternion(pso[0:4])
    # angles_right_knee = pso[4]
    # angle_left_knee = pso[13]
    # print("angles_chest", angles_chest) 
    # print("angles_lhip", angles_left_hip) 
    # print("angles rhip", angles_right_hip) 
    # print("angles_rknee", angles_right_knee)
    # print("angles_lknee", angle_left_knee)
    # print("\n")
    p.stepSimulation()
    time.sleep(1. / 240.)





