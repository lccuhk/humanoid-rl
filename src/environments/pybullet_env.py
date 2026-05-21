"""PyBullet Environment for Humanoid Robot Locomotion"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


class HumanoidPyBulletEnv(gym.Env):
    """
    Custom PyBullet environment for humanoid robot walking and running.
    
    This environment simulates a humanoid robot learning to walk and run
    using deep reinforcement learning with PyBullet physics engine.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        model_path: Optional[str] = None,
        task: str = "walk",
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 5.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.8, 2.0),
        reset_noise_scale: float = 0.01,
        exclude_current_positions_from_observation: bool = True,
    ):
        """
        Initialize the humanoid PyBullet environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array')
            model_path: Path to custom URDF model file
            task: Task type ('walk' or 'run')
            forward_reward_weight: Weight for forward velocity reward
            ctrl_cost_weight: Weight for control cost penalty
            contact_cost_weight: Weight for contact force penalty
            healthy_reward: Reward for staying healthy
            terminate_when_unhealthy: Whether to terminate when unhealthy
            healthy_z_range: Valid z-height range for the robot
            reset_noise_scale: Scale of random noise on reset
            exclude_current_positions_from_observation: Whether to exclude x/y positions
        """
        super().__init__()
        
        if not PYBULLET_AVAILABLE:
            raise ImportError(
                "PyBullet is not installed. Please install it with: pip install pybullet"
            )
        
        self.render_mode = render_mode
        self.task = task
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.contact_cost_weight = contact_cost_weight
        self.healthy_reward = healthy_reward
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self.healthy_z_range = healthy_z_range
        self.reset_noise_scale = reset_noise_scale
        self.exclude_current_positions_from_observation = exclude_current_positions_from_observation
        
        self.model_path = model_path
        self.physics_client = None
        self.robot_id = None
        self.plane_id = None
        self._initialize_simulation()
        self._setup_action_space()
        self._setup_observation_space()
    
    def _initialize_simulation(self):
        """Initialize PyBullet simulation."""
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        
        self.plane_id = p.loadURDF("plane.urdf")
        
        if self.model_path is None:
            self.robot_id = p.loadURDF(
                "humanoid.urdf",
                [0, 0, 1.2],
                p.getQuaternionFromEuler([0, 0, 0])
            )
        else:
            self.robot_id = p.loadURDF(
                self.model_path,
                [0, 0, 1.2],
                p.getQuaternionFromEuler([0, 0, 0])
            )
        
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = []
        self.joint_names = []
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_names.append(joint_info[1].decode('utf-8'))
        
        self.num_controllable_joints = len(self.joint_indices)
        
        for idx in self.joint_indices:
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                p.VELOCITY_CONTROL,
                force=0
            )
    
    def _setup_action_space(self):
        """Setup action space."""
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_controllable_joints,),
            dtype=np.float32
        )
    
    def _setup_observation_space(self):
        """Setup observation space."""
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        
        obs_dim = 0
        if self.exclude_current_positions_from_observation:
            obs_dim += 1
        else:
            obs_dim += 3
        
        obs_dim += 3
        obs_dim += 3
        obs_dim += 3
        
        for idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, idx)
            obs_dim += 2
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        base_orn_euler = p.getEulerFromQuaternion(base_orn)
        
        observation = []
        
        if self.exclude_current_positions_from_observation:
            observation.append(base_pos[2])
        else:
            observation.extend(base_pos)
        
        observation.extend(base_orn_euler)
        observation.extend(base_lin_vel)
        observation.extend(base_ang_vel)
        
        for idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, idx)
            observation.append(joint_state[0])
            observation.append(joint_state[1])
        
        return np.array(observation, dtype=np.float32)
    
    def _is_healthy(self) -> bool:
        """Check if robot is healthy."""
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        z = base_pos[2]
        return self.healthy_z_range[0] < z < self.healthy_z_range[1]
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        action = np.clip(action, -1.0, 1.0)
        
        max_force = 100.0
        for i, idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                p.TORQUE_CONTROL,
                force=action[i] * max_force
            )
        
        for _ in range(4):
            p.stepSimulation()
        
        observation = self._get_observation()
        
        healthy = self._is_healthy()
        terminated = not healthy if self.terminate_when_unhealthy else False
        truncated = False
        
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, _ = p.getBaseVelocity(self.robot_id)
        forward_vel = base_lin_vel[0]
        
        forward_reward = self.forward_reward_weight * forward_vel
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        
        contact_cost = 0
        if self.contact_cost_weight > 0:
            contact_points = p.getContactPoints(self.robot_id)
            if contact_points:
                contact_forces = [cp[9] for cp in contact_points]
                contact_cost = self.contact_cost_weight * np.sum(np.square(contact_forces))
        
        healthy_reward = self.healthy_reward if healthy else 0
        
        if self.task == "run":
            forward_reward *= 1.5
        
        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        
        info = {
            'reward_forward': forward_reward,
            'reward_healthy': healthy_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'x_velocity': forward_vel,
            'z_position': base_pos[2],
            'task': self.task,
            'is_healthy': healthy
        }
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [0, 0, 1.2],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
        
        for idx in self.joint_indices:
            noise = self.np_random.uniform(
                low=-self.reset_noise_scale,
                high=self.reset_noise_scale
            )
            p.resetJointState(self.robot_id, idx, noise, 0)
        
        observation = self._get_observation()
        info = {
            'task': self.task,
            'is_healthy': self._is_healthy()
        }
        
        return observation, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        if self.render_mode == "rgb_array":
            width, height = 640, 480
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 1],
                distance=3,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(width) / height,
                nearVal=0.1,
                farVal=100.0
            )
            _, _, px, _, _ = p.getCameraImage(
                width, height,
                view_matrix, proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (height, width, 4))
            return rgb_array[:, :, :3]
        
        return None
    
    def close(self):
        """Close the environment."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for logging/visualization."""
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        base_orn_euler = p.getEulerFromQuaternion(base_orn)
        
        joint_positions = []
        joint_velocities = []
        for idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, idx)
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])
        
        return {
            'base_position': np.array(base_pos),
            'base_orientation': np.array(base_orn_euler),
            'base_linear_velocity': np.array(base_lin_vel),
            'base_angular_velocity': np.array(base_ang_vel),
            'joint_positions': np.array(joint_positions),
            'joint_velocities': np.array(joint_velocities),
            'x_position': base_pos[0],
            'y_position': base_pos[1],
            'z_position': base_pos[2],
            'x_velocity': base_lin_vel[0],
            'is_healthy': self._is_healthy()
        }
