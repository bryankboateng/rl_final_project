from typing import Optional, Tuple, Any
import numpy as np
from simulators.pybullet_debugger import pybulletDebug
from .resources.plane import Plane
from .base_dynamics import BaseDynamics
import pybullet as p
import subprocess


class BasePybulletDynamics(BaseDynamics):

  def __init__(self, config: Any, action_space: np.ndarray) -> None:
    super().__init__(config, action_space)
    """
    Initialize a Pybullet physics simulator to keep track of robot dynamics. This will only work for single agent

    Args:
        config (Any): an object specifies configuration. This will correspond to config_agent of the yaml config file
        action_space (np.ndarray): action space.
    """

    self.verbose = config.verbose
    self.gui = config.gui
    self.gui_imaginary = config.gui_imaginary
    self.dt = config.dt
    self.gravity = -9.81

    # configure force in the dynamics
    self.replace_adv_with_dr = False
    if config.replace_adv_with_dr:
      self.replace_adv_with_dr = config.replace_adv_with_dr

    if config.apply_force:
      self.force = float(config.force) * float(config.force_scale)
    else:
      self.force = 0.0
    self.force_info = config.force_info
    self.elapsed_force_applied = 0
    self.force_applied_reset = config.force_reset_time

    self.rotate_reset = config.rotate_reset
    self.height_reset = config.height_reset  # drop, stand, both

    self.force_applied_force_vector = None
    self.force_applied_position_vector = None
    self.force_type = config.force_type
    self.link_name = config.link_name

    # configure terrain in the dynamics
    self.terrain = config.terrain
    self.terrain_height = config.terrain_height
    self.terrain_gridsize = config.terrain_gridsize
    self.terrain_friction = config.terrain_friction

    self.terrain_data = None

    self.reset_criterion = config.reset_criterion

    # initialize a pybullet client (GUI/DIRECT)
    if self.gui:
      # Setup the GUI (disable the useless windows)
      self.camera_info = {'camera': {'distance': 12, 'yaw': -0, 'pitch': -89}, 'lookat': [0, 0, 0]}
      self._render_width = 640
      self._render_height = 480
      self.client = p.connect(p.GUI)
      p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
      p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
      p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
      p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
      # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
      p.resetDebugVisualizerCamera(
          cameraDistance=1, cameraYaw=20, cameraPitch=-20, cameraTargetPosition=[1, -0.5, 0.8],
          physicsClientId=self.client
      )
      self.debugger = pybulletDebug(self.client)
    else:
      self.client = p.connect(p.DIRECT)

    self.video_output_file = None
    self.ffmpeg_pipe = None
    self.cnt = None

  def _init_frames(self):
    """
    Initialize the pipe for streaming frames to the video file.
    Warning: video slows down the simulation!
    """
    if self.ffmpeg_pipe is not None:
      try:
        if self.video_output_file is not None:
          self.ffmpeg_pipe.stdin.close()
          self.ffmpeg_pipe.stderr.close()
          ret = self.ffmpeg_pipe.wait()
      except Exception as e:
        print("VideoRecorder encoder exited with status {}".format(ret))

    if self.video_output_file is not None:
      camera = p.getDebugVisualizerCamera()
      command = [
          'ffmpeg', '-y', '-r',
          str(24), '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '{}x{}'.format(camera[0], camera[1]), '-pix_fmt',
          'rgba', '-i', '-', '-an', '-vcodec', 'mpeg4', '-vb', '20M', self.video_output_file
      ]
      self.ffmpeg_pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

  def _save_frames(self):
    """
    Write frame at each step.
    24 FPS dt = 1/240 : every 10 frames
    """
    # if self.video_output_file is not None and self.cnt % (int(1. / (self.dt * 24))) == 0:
    if self.video_output_file is not None:
      camera = p.getDebugVisualizerCamera(physicsClientId=self.client)
      img = p.getCameraImage(camera[0], camera[1], renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.client)
      self.ffmpeg_pipe.stdin.write(img[2].tobytes())

  def destroy(self):
    """Properly close the simulation."""
    try:
      p.disconnect()
    except p.error as e:
      print("Warning (destructor of simulator):", e)

    self.close_video_stream()

  def close_video_stream(self):
    try:
      if self.gui and self.video_output_file is not None:
        self.ffmpeg_pipe.stdin.close()
        self.ffmpeg_pipe.stderr.close()
        ret = self.ffmpeg_pipe.wait()
    except Exception as e:
      print("VideoRecorder encoder exited with status {}".format(ret))

  def reset(self, **kwargs):
    if "video_output_file" in kwargs.keys():
      self.video_output_file = kwargs["video_output_file"]

    if "adversarial_sequence" in kwargs.keys():
      self.adversarial_sequence = kwargs["adversarial_sequence"]
    else:
      # clean adversarial sequence, so that any adversarial sequence passed during a reset will only be used once
      self.adversarial_sequence = None

    self.cnt = 0
    p.resetSimulation(physicsClientId=self.client)
    p.setGravity(0, 0, self.gravity, physicsClientId=self.client)
    p.setTimeStep(self.dt, physicsClientId=self.client)
    p.setPhysicsEngineParameter(fixedTimeStep=self.dt, physicsClientId=self.client)
    p.setRealTimeSimulation(0, physicsClientId=self.client)
    Plane(self.client)

    if "terrain_data" in kwargs.keys():
      terrain_data = kwargs["terrain_data"]
    else:
      terrain_data = None

    if self.terrain == "rough":
      if terrain_data is None:
        self._gen_terrain(mesh_scale=[self.terrain_gridsize, self.terrain_gridsize, 2.0])
      else:
        self._set_terrain(terrain_data, mesh_scale=[self.terrain_gridsize, self.terrain_gridsize, 2.0])

    self._gen_force()

    if self.gui and self.video_output_file is not None:
      self._init_frames()

  def integrate_forward(
      self, state: np.ndarray, control: np.ndarray, num_segment: Optional[int] = 1, noise: Optional[np.ndarray] = None,
      noise_type: Optional[str] = 'unif', adversary: Optional[np.ndarray] = None, **kwargs
  ) -> Tuple[np.ndarray, np.ndarray]:
    return super().integrate_forward(state, control, num_segment, noise, noise_type, adversary, **kwargs)

  def get_jacobian(self, nominal_states: np.ndarray, nominal_controls: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return super().get_jacobian(nominal_states, nominal_controls)

  def _gen_force(self):
    """
    Create a random force to be applied onto the robot
    The force will be applied when integrate_forward is called
    """
    # create a random force applied on the robot
    self.elapsed_force_applied = 0
    if self.force_info != None:
      self.force_applied_force_vector = np.array([
          np.random.uniform(self.force_info["vector"][0][0], self.force_info["vector"][0][1]),
          np.random.uniform(self.force_info["vector"][1][0], self.force_info["vector"][1][1]),
          np.random.uniform(self.force_info["vector"][2][0], self.force_info["vector"][2][1])
      ])
      self.force_applied_position_vector = np.array([
          np.random.uniform(self.force_info["position"][0][0], self.force_info["position"][0][1]),
          np.random.uniform(self.force_info["position"][1][0], self.force_info["position"][1][1]),
          np.random.uniform(self.force_info["position"][2][0], self.force_info["position"][2][1])
      ])
    elif self.force_type == "uniform":
      self.force_applied_force_vector = np.array([
          np.random.uniform(-1, 1), np.random.uniform(-1, 1),
          np.random.uniform(-1, 1)
      ])
      self.force_applied_position_vector = np.array([
          np.random.uniform(-0.1, 0.1),
          np.random.uniform(-0.1, 0.1),
          np.random.uniform(-0.05, 0.05)
      ])
    elif self.force_type == "bangbang":
      self.force_applied_force_vector = np.array([
          np.random.choice([-1, 0, 1], p=[1 / 3, 1 / 3, 1 / 3]),
          np.random.choice([-1, 0, 1], p=[1 / 3, 1 / 3, 1 / 3]),
          np.random.choice([-1, 0, 1], p=[1 / 3, 1 / 3, 1 / 3])
      ])
      self.force_applied_position_vector = np.array([
          np.random.choice([-0.1, 0, 0.1], p=[1 / 3, 1 / 3, 1 / 3]),
          np.random.choice([-0.1, 0, 0.1], p=[1 / 3, 1 / 3, 1 / 3]),
          np.random.choice([-0.05, 0, 0.05], p=[1 / 3, 1 / 3, 1 / 3])
      ])

    else:
      self.force_applied_force_vector = np.array([
          np.random.uniform(-1, 1), np.random.uniform(-1, 1),
          np.random.uniform(-50, 5)
      ])
      self.force_applied_position_vector = np.array([
          np.random.uniform(-0.1, 0.1),
          np.random.uniform(-0.1, 0.1),
          np.random.uniform(-0.05, 0.05)
      ])

    # normalize force (bang bang)
    two_norm = np.linalg.norm(self.force_applied_force_vector, 2)
    if two_norm != 0:
      normed_force_vector = self.force_applied_force_vector / two_norm
      self.force_applied_force_vector = normed_force_vector * self.force
    else:
      self.force_applied_force_vector = self.force_applied_force_vector * self.force

  def _apply_adversarial_force(self, force_vector, position_vector):
    two_norm = np.linalg.norm(force_vector, 2)
    if two_norm != 0:
      normed_force_vector = force_vector / two_norm
      self.force_applied_force_vector = normed_force_vector * self.force
    else:
      self.force_applied_force_vector = force_vector * self.force

    # print(np.linalg.norm(self.force_applied_force_vector))

    self.force_applied_position_vector = position_vector

    if self.link_name is not None:
      p.applyExternalForce(
          self.robot.id, self.robot.get_link_id(self.link_name), self.force_applied_force_vector,
          self.force_applied_position_vector, p.LINK_FRAME, physicsClientId=self.client
      )
    else:
      p.applyExternalForce(
          self.robot.id, -1, self.force_applied_force_vector, self.force_applied_position_vector, p.LINK_FRAME,
          physicsClientId=self.client
      )

  def _apply_force(self):
    if self.elapsed_force_applied > self.force_applied_reset:
      self._gen_force()
    else:
      self.elapsed_force_applied += 1

    if self.adv_debug_line_id is not None:
      p.removeUserDebugItem(self.adv_debug_line_id)

    if self.link_name is not None:
      p.applyExternalForce(
          self.robot.id, self.robot.get_link_id(self.link_name), self.force_applied_force_vector,
          self.force_applied_position_vector, p.LINK_FRAME, physicsClientId=self.client
      )
      self.adv_debug_line_id = p.addUserDebugLine(
          self.force_applied_position_vector, self.force_applied_position_vector + self.force_applied_force_vector,
          lineColorRGB=[0, 0, 1], lineWidth=2.0, physicsClientId=self.client, parentObjectUniqueId=self.robot.id,
          parentLinkIndex=self.robot.get_link_id(self.link_name)
      )
    else:
      p.applyExternalForce(
          self.robot.id, -1, self.force_applied_force_vector, self.force_applied_position_vector, p.LINK_FRAME,
          physicsClientId=self.client
      )
      self.adv_debug_line_id = p.addUserDebugLine(
          self.force_applied_position_vector, self.force_applied_position_vector + self.force_applied_force_vector,
          lineColorRGB=[0, 0, 1], lineWidth=2.0, physicsClientId=self.client, parentObjectUniqueId=self.robot.id
      )

  def _apply_dstb_from_adversarial_sequence(self):
    if self.adversarial_sequence is None:
      return False
    try:
      self.force = self.adversarial_sequence[self.cnt]["force"]
      self.force_applied_force_vector = self.adversarial_sequence[self.cnt]["force_applied_force_vector"]
      self.force_applied_position_vector = self.adversarial_sequence[self.cnt]["force_applied_position_vector"]
    except IndexError:
      return False

    if self.adv_debug_line_id is not None:
      p.removeUserDebugItem(self.adv_debug_line_id)

    if self.link_name is not None:
      p.applyExternalForce(
          self.robot.id, self.robot.get_link_id(self.link_name), self.force_applied_force_vector,
          self.force_applied_position_vector, p.LINK_FRAME, physicsClientId=self.client
      )
      self.adv_debug_line_id = p.addUserDebugLine(
          self.force_applied_position_vector, self.force_applied_position_vector + self.force_applied_force_vector,
          lineColorRGB=[0, 0, 1], lineWidth=2.0, physicsClientId=self.client, parentObjectUniqueId=self.robot.id,
          parentLinkIndex=self.robot.get_link_id(self.link_name)
      )
    else:
      p.applyExternalForce(
          self.robot.id, -1, self.force_applied_force_vector, self.force_applied_position_vector, p.LINK_FRAME,
          physicsClientId=self.client
      )
      self.adv_debug_line_id = p.addUserDebugLine(
          self.force_applied_position_vector, self.force_applied_position_vector + self.force_applied_force_vector,
          lineColorRGB=[0, 0, 1], lineWidth=2.0, physicsClientId=self.client, parentObjectUniqueId=self.robot.id
      )

    return True

  def _gen_terrain(self, terrain_height: Optional[int] = None, mesh_scale: Optional[np.ndarray] = None):
    """
    Create a randomized terrain to be applied into the dynamics
    The terrain will be applied from the beginning
    """
    if terrain_height is not None:
      self.terrain_height = terrain_height

    if mesh_scale is None:
      mesh_scale = [0.08, 0.08, 1.0]  # [x, y, z]

    heightPerturbationRange = self.terrain_height
    numHeightfieldRows = 256
    numHeightfieldColumns = 256

    terrainShape = 0
    heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns

    heightPerturbationRange = heightPerturbationRange
    for j in range(int(numHeightfieldColumns / 2)):
      for i in range(int(numHeightfieldRows / 2)):
        height = np.random.uniform(0, heightPerturbationRange)
        heightfieldData[2*i + 2*j*numHeightfieldRows] = height
        heightfieldData[2*i + 1 + 2*j*numHeightfieldRows] = height
        heightfieldData[2*i + (2*j + 1) * numHeightfieldRows] = height
        heightfieldData[2*i + 1 + (2*j + 1) * numHeightfieldRows] = height

    terrainShape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        # meshScale=[0.08, 0.08, 1.0], # [x, y, z]
        meshScale=mesh_scale,
        heightfieldTextureScaling=(numHeightfieldRows-1) / 2,
        heightfieldData=heightfieldData,
        numHeightfieldRows=numHeightfieldRows,
        numHeightfieldColumns=numHeightfieldColumns,
        physicsClientId=self.client
    )

    terrain = p.createMultiBody(0, terrainShape, physicsClientId=self.client)

    p.resetBasePositionAndOrientation(terrain, [0, 0, 0.0], [0, 0, 0, 1], physicsClientId=self.client)

    p.changeDynamics(terrain, -1, lateralFriction=self.terrain_friction, physicsClientId=self.client)
    p.changeVisualShape(terrain, -1, rgbaColor=[0.2, 0.8, 0.8, 1], physicsClientId=self.client)

    self.terrain_data = heightfieldData

  def _set_terrain(self, terrain_data, terrain_height: Optional[int] = None, mesh_scale: Optional[np.ndarray] = None):
    """
        Create a randomized terrain to be applied into the dynamics
        The terrain will be applied from the beginning
        """
    if terrain_height is not None:
      self.terrain_height = terrain_height

    if mesh_scale is None:
      mesh_scale = [0.08, 0.08, 1.0]  # [x, y, z]

    numHeightfieldRows = 256
    numHeightfieldColumns = 256

    terrainShape = 0
    heightfieldData = terrain_data

    terrainShape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD, meshScale=mesh_scale, heightfieldTextureScaling=(numHeightfieldRows-1) / 2,
        heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows,
        numHeightfieldColumns=numHeightfieldColumns, physicsClientId=self.client
    )

    terrain = p.createMultiBody(0, terrainShape, physicsClientId=self.client)

    p.resetBasePositionAndOrientation(terrain, [0, 0, 0.0], [0, 0, 0, 1], physicsClientId=self.client)

    p.changeDynamics(terrain, -1, lateralFriction=self.terrain_friction, physicsClientId=self.client)
    p.changeVisualShape(terrain, -1, rgbaColor=[0.2, 0.8, 0.8, 1], physicsClientId=self.client)

    self.terrain_data = heightfieldData
