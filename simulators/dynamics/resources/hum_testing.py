import pybullet as p
import time
from math import pi
from humanoid import Humanoid  # make sure this is your correct import

if __name__ == "__main__":
    client = p.connect(p.GUI)
    time.sleep(1.0)  # Allow GUI to fully initialize

    humanoid = Humanoid(client)

    p.resetDebugVisualizerCamera(
        cameraDistance=10,
        cameraYaw=45,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 1]
    )

    humanoid.reset([0, 0, 1.5])
    humanoid.apply_action([0.1] * 10)

    joint_sliders = {}  # {joint_name: slider or (roll, pitch, yaw)}
    joint_type_map = {}

    for idx in humanoid.joint_index:
        joint_info = p.getJointInfo(humanoid.id, idx)
        joint_name = joint_info[1].decode()
        lo, hi = joint_info[8], joint_info[9]
        joint_type = joint_info[2]
        joint_type_map[joint_name] = joint_type

        if joint_type == p.JOINT_FIXED or joint_type == p.JOINT_SPHERICAL:
            print(f"Skipping fixed joint: {joint_name}")
            continue

        # if joint_type == p.JOINT_SPHERICAL:
        #     # Add 3 sliders for roll, pitch, yaw (in radians)
        #     # joint_sliders[joint_name] = (
        #     #     p.addUserDebugParameter(f"{joint_name}_roll", -pi, pi, 0.0),
        #     #     p.addUserDebugParameter(f"{joint_name}_pitch", -pi, pi, 0.0),
        #     #     p.addUserDebugParameter(f"{joint_name}_yaw", -pi, pi, 0.0)
        #     # )
        #     continue
        else:
            # Add 1D slider for revolute/prismatic joints
            if hi > lo:
                slider = p.addUserDebugParameter(joint_name, lo, hi, 0.0)
            else:
                slider = p.addUserDebugParameter(joint_name, -pi, pi, 0.0)
            joint_sliders[joint_name] = slider

    p.resetBasePositionAndOrientation(
        humanoid.id, [0, 0, 4], p.getQuaternionFromEuler([1.57, 0, 0])
    )

    for _ in range(240):
        p.stepSimulation()
        time.sleep(1. / 240.)

    while p.isConnected():
        target_angles = []
        for joint_name in joint_sliders:  # ensure this matches your joint order
            joint_type = joint_type_map.get(joint_name)
            slider = joint_sliders.get(joint_name)

            if joint_type == p.JOINT_SPHERICAL and slider or joint_type == p.JOINT_FIXED:
                # roll = p.readUserDebugParameter(slider[0])
                # pitch = p.readUserDebugParameter(slider[1])
                # yaw = p.readUserDebugParameter(slider[2])
                # quat = p.getQuaternionFromEuler([roll, pitch, yaw])
                # target_angles.append(quat)
                continue
            elif slider is not None:
                angle = p.readUserDebugParameter(slider)
                target_angles.append(angle)
            else:
                target_angles.append(0.0)  # default
        print(len(target_angles))
        humanoid.apply_position(target_angles)
        p.stepSimulation()
        time.sleep(1. / 240.)
