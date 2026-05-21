"""Isaac Gym Environment for Humanoid Robot Locomotion"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional

try:
    import torch
    import isaacgym
    from isaacgym import gymapi
    from isaacgym import gymutil
    ISAAC_GYM_AVAILABLE = True
except ImportError:
    ISAAC_GYM_AVAILABLE = False


class HumanoidIsaacGymEnv(gym.Env):
    """
    Custom Isaac Gym environment for humanoid robot walking and running.
    
    This environment simulates a humanoid robot learning to walk and run
    using deep reinforcement learning with NVIDIA Isaac Gym for GPU-accelerated
    physics simulation.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_envs: int = 1,
        task: str = "walk",
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 5.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.8, 2.0),
        reset_noise_scale: float = 0.01,
        device: str = "cuda",
    ):
        """
        Initialize the humanoid Isaac Gym environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array')
            num_envs: Number of parallel environments
            task: Task type ('walk' or 'run')
            forward_reward_weight: Weight for forward velocity reward
            ctrl_cost_weight: Weight for control cost penalty
            contact_cost_weight: Weight for contact force penalty
            healthy_reward: Reward for staying healthy
            terminate_when_unhealthy: Whether to terminate when unhealthy
            healthy_z_range: Valid z-height range for the robot
            reset_noise_scale: Scale of random noise on reset
            device: Device to use ('cuda' or 'cpu')
        """
        super().__init__()
        
        if not ISAAC_GYM_AVAILABLE:
            raise ImportError(
                "Isaac Gym is not installed. Please install it from NVIDIA's website."
            )
        
        self.render_mode = render_mode
        self.num_envs = num_envs
        self.task = task
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.contact_cost_weight = contact_cost_weight
        self.healthy_reward = healthy_reward
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self.healthy_z_range = healthy_z_range
        self.reset_noise_scale = reset_noise_scale
        self.device = device
        
        self.gym = None
        self.sim = None
        self.envs = []
        self.actor_handles = []
        self._initialize_simulation()
        self._setup_action_space()
        self._setup_observation_space()
    
    def _initialize_simulation(self):
        """Initialize Isaac Gym simulation."""
        self.gym = gymapi.acquire_gym()
        
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        if self.device == "cuda":
            sim_params.physx.use_gpu = True
            sim_params.use_gpu_pipeline = True
        
        graphics_device_id = 0 if self.render_mode == "human" else -1
        self.sim = self.gym.create_sim(0, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        asset_file = "humanoid.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.armature = 0.01
        
        try:
            self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        except:
            print("Warning: Could not load custom humanoid asset, using simple box as placeholder")
            self.robot_asset = self._create_simple_asset()
        
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env)
            
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(0.0, 0.0, 1.2)
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            actor_handle = self.gym.create_actor(env, self.robot_asset, start_pose, "humanoid", i, 1)
            self.actor_handles.append(actor_handle)
            
            dof_props = self.gym.get_actor_dof_properties(env, actor_handle)
            dof_props['stiffness'].fill(1000.0)
            dof_props['damping'].fill(100.0)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)
        
        if self.render_mode == "human":
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            cam_pos = gymapi.Vec3(3.0, 0.0, 2.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        self.gym.prepare_sim(self.sim)
    
    def _create_simple_asset(self):
        """Create a simple humanoid asset."""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        
        asset = self.gym.create_asset_box(self.sim, 0.5, 0.5, 1.0, asset_options)
        return asset
    
    def _setup_action_space(self):
        """Setup action space."""
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_dof,),
            dtype=np.float32
        )
    
    def _setup_observation_space(self):
        """Setup observation space."""
        obs_dim = self.num_dof * 2 + 13
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def _get_observation(self, env_idx: int = 0) -> np.ndarray:
        """Get current observation for a specific environment."""
        env = self.envs[env_idx]
        actor_handle = self.actor_handles[env_idx]
        
        rigid_body_state = self.gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)
        dof_state = self.gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
        
        base_pos = rigid_body_state['pose']['p'][0]
        base_rot = rigid_body_state['pose']['r'][0]
        base_lin_vel = rigid_body_state['vel']['linear'][0]
        base_ang_vel = rigid_body_state['vel']['angular'][0]
        
        dof_pos = dof_state['pos']
        dof_vel = dof_state['vel']
        
        observation = np.concatenate([
            [base_pos[2]],
            [base_rot[0], base_rot[1], base_rot[2], base_rot[3]],
            [base_lin_vel[0], base_lin_vel[1], base_lin_vel[2]],
            [base_ang_vel[0], base_ang_vel[1], base_ang_vel[2]],
            dof_pos,
            dof_vel
        ])
        
        return observation.astype(np.float32)
    
    def _is_healthy(self, env_idx: int = 0) -> bool:
        """Check if robot is healthy."""
        env = self.envs[env_idx]
        actor_handle = self.actor_handles[env_idx]
        
        rigid_body_state = self.gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
        z = rigid_body_state['pose']['p'][0][2]
        
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
        
        for i in range(self.num_envs):
            env = self.envs[i]
            actor_handle = self.actor_handles[i]
            
            dof_props = self.gym.get_actor_dof_properties(env, actor_handle)
            effort_limits = dof_props['effort']
            
            efforts = action * effort_limits
            self.gym.set_actor_dof_efforts(env, actor_handle, efforts)
        
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        if self.render_mode == "human":
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
        
        observation = self._get_observation(0)
        
        healthy = self._is_healthy(0)
        terminated = not healthy if self.terminate_when_unhealthy else False
        truncated = False
        
        env = self.envs[0]
        actor_handle = self.actor_handles[0]
        rigid_body_state = self.gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)
        
        base_pos = rigid_body_state['pose']['p'][0]
        base_lin_vel = rigid_body_state['vel']['linear'][0]
        forward_vel = base_lin_vel[0]
        
        forward_reward = self.forward_reward_weight * forward_vel
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        
        contact_cost = 0
        if self.contact_cost_weight > 0:
            net_contact_forces = self.gym.get_net_contact_forces(env, actor_handle)
            contact_cost = self.contact_cost_weight * np.sum(np.square(net_contact_forces))
        
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
        
        for i in range(self.num_envs):
            env = self.envs[i]
            actor_handle = self.actor_handles[i]
            
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(0.0, 0.0, 1.2)
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            self.gym.set_actor_root_state(env, actor_handle, gymapi.Transform(start_pose.p, start_pose.r), gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0))
            
            dof_state = self.gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
            noise = self.np_random.uniform(
                low=-self.reset_noise_scale,
                high=self.reset_noise_scale,
                size=dof_state.shape
            )
            dof_state['pos'] += noise
            dof_state['vel'] = 0
            self.gym.set_actor_dof_states(env, actor_handle, dof_state, gymapi.STATE_ALL)
        
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        observation = self._get_observation(0)
        info = {
            'task': self.task,
            'is_healthy': self._is_healthy(0)
        }
        
        return observation, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        if self.render_mode == "rgb_array":
            return None
        
        return None
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for logging/visualization."""
        env = self.envs[0]
        actor_handle = self.actor_handles[0]
        
        rigid_body_state = self.gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)
        dof_state = self.gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
        
        base_pos = rigid_body_state['pose']['p'][0]
        base_rot = rigid_body_state['pose']['r'][0]
        base_lin_vel = rigid_body_state['vel']['linear'][0]
        base_ang_vel = rigid_body_state['vel']['angular'][0]
        
        return {
            'base_position': np.array([base_pos[0], base_pos[1], base_pos[2]]),
            'base_orientation': np.array([base_rot[0], base_rot[1], base_rot[2], base_rot[3]]),
            'base_linear_velocity': np.array([base_lin_vel[0], base_lin_vel[1], base_lin_vel[2]]),
            'base_angular_velocity': np.array([base_ang_vel[0], base_ang_vel[1], base_ang_vel[2]]),
            'joint_positions': dof_state['pos'].copy(),
            'joint_velocities': dof_state['vel'].copy(),
            'x_position': base_pos[0],
            'y_position': base_pos[1],
            'z_position': base_pos[2],
            'x_velocity': base_lin_vel[0],
            'is_healthy': self._is_healthy(0)
        }
