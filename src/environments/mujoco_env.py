"""MuJoCo Environment for Humanoid Robot Locomotion"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
from typing import Tuple, Dict, Any, Optional


class HumanoidMuJoCoEnv(gym.Env):
    """
    Custom MuJoCo environment for humanoid robot walking and running.
    
    This environment simulates a humanoid robot learning to walk and run
    using deep reinforcement learning.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        model_path: Optional[str] = None,
        task: str = "walk",
        forward_reward_weight: float = 5.0,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.8, 2.0),
        reset_noise_scale: float = 0.01,
        exclude_current_positions_from_observation: bool = True,
        backward_penalty_weight: float = 10.0,
        position_reward_weight: float = 2.0,
        velocity_bonus_weight: float = 30.0,
        air_time_reward_weight: float = 50.0,
        energy_efficiency_weight: float = 5.0,
    ):
        """
        Initialize the humanoid MuJoCo environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', 'depth_array')
            model_path: Path to custom MuJoCo model XML file
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
        self.backward_penalty_weight = backward_penalty_weight
        self.position_reward_weight = position_reward_weight
        self.velocity_bonus_weight = velocity_bonus_weight
        self.air_time_reward_weight = air_time_reward_weight
        self.energy_efficiency_weight = energy_efficiency_weight
        self.start_x_position = 0.0
        self.air_time = 0.0
        self.last_foot_contact = True
        
        if model_path is None:
            self.model_path = self._get_default_model_path()
        else:
            self.model_path = model_path
        
        self._initialize_model()
        self._setup_action_space()
        self._setup_observation_space()
        
        self.viewer = None
        self.camera = None
    
    def _get_default_model_path(self) -> str:
        """Get path to default humanoid model."""
        try:
            import gymnasium.envs.mujoco.humanoid_v5 as humanoid_v5
            return humanoid_v5.DEFAULT_CAMERA_CONFIG
        except:
            return None
    
    def _initialize_model(self):
        """Initialize MuJoCo model and data."""
        try:
            self.env = gym.make(
                "Humanoid-v5",
                render_mode=self.render_mode,
                forward_reward_weight=self.forward_reward_weight,
                ctrl_cost_weight=self.ctrl_cost_weight,
                contact_cost_weight=self.contact_cost_weight,
                healthy_reward=self.healthy_reward,
                terminate_when_unhealthy=self.terminate_when_unhealthy,
                healthy_z_range=self.healthy_z_range,
                reset_noise_scale=self.reset_noise_scale,
                exclude_current_positions_from_observation=self.exclude_current_positions_from_observation,
            )
            self.model = self.env.unwrapped.model
            self.data = self.env.unwrapped.data
        except Exception as e:
            print(f"Warning: Could not create Humanoid-v5 environment: {e}")
            print("Using fallback simple humanoid model...")
            self._create_simple_model()
    
    def _create_simple_model(self):
        """Create a simple humanoid model as fallback."""
        xml = """
        <mujoco model="simple_humanoid">
            <compiler angle="degree" inertiafromgeom="true"/>
            <option timestep="0.01" gravity="0 0 -9.81"/>
            <worldbody>
                <light name="light1" pos="0 0 3" dir="0 0 -1"/>
                <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.9 0.8 1"/>
                <body name="torso" pos="0 0 1.2">
                    <joint name="root" type="free"/>
                    <geom name="torso" type="capsule" fromto="0 0 0 0 0 0.3" size="0.2" rgba="0.8 0.3 0.3 1"/>
                    <body name="head" pos="0 0 0.4">
                        <geom name="head" type="sphere" size="0.15" rgba="0.8 0.3 0.3 1"/>
                    </body>
                    <body name="left_arm" pos="0.2 0 0.2">
                        <joint name="left_shoulder" type="hinge" axis="0 1 0"/>
                        <geom name="left_upper_arm" type="capsule" fromto="0 0 0 0.3 0 0" size="0.08" rgba="0.3 0.3 0.8 1"/>
                        <body name="left_forearm" pos="0.3 0 0">
                            <joint name="left_elbow" type="hinge" axis="0 1 0"/>
                            <geom name="left_forearm" type="capsule" fromto="0 0 0 0.25 0 0" size="0.07" rgba="0.3 0.3 0.8 1"/>
                        </body>
                    </body>
                    <body name="right_arm" pos="-0.2 0 0.2">
                        <joint name="right_shoulder" type="hinge" axis="0 1 0"/>
                        <geom name="right_upper_arm" type="capsule" fromto="0 0 0 -0.3 0 0" size="0.08" rgba="0.3 0.3 0.8 1"/>
                        <body name="right_forearm" pos="-0.3 0 0">
                            <joint name="right_elbow" type="hinge" axis="0 1 0"/>
                            <geom name="right_forearm" type="capsule" fromto="0 0 0 -0.25 0 0" size="0.07" rgba="0.3 0.3 0.8 1"/>
                        </body>
                    </body>
                    <body name="left_leg" pos="0.1 0 -0.2">
                        <joint name="left_hip" type="hinge" axis="0 1 0"/>
                        <geom name="left_thigh" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.1" rgba="0.3 0.8 0.3 1"/>
                        <body name="left_shin" pos="0 0 -0.4">
                            <joint name="left_knee" type="hinge" axis="0 1 0"/>
                            <geom name="left_shin" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.09" rgba="0.3 0.8 0.3 1"/>
                            <body name="left_foot" pos="0 0 -0.4">
                                <geom name="left_foot" type="box" size="0.12 0.08 0.05" rgba="0.3 0.8 0.3 1"/>
                            </body>
                        </body>
                    </body>
                    <body name="right_leg" pos="-0.1 0 -0.2">
                        <joint name="right_hip" type="hinge" axis="0 1 0"/>
                        <geom name="right_thigh" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.1" rgba="0.3 0.8 0.3 1"/>
                        <body name="right_shin" pos="0 0 -0.4">
                            <joint name="right_knee" type="hinge" axis="0 1 0"/>
                            <geom name="right_shin" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.09" rgba="0.3 0.8 0.3 1"/>
                            <body name="right_foot" pos="0 0 -0.4">
                                <geom name="right_foot" type="box" size="0.12 0.08 0.05" rgba="0.3 0.8 0.3 1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <motor name="left_shoulder_motor" joint="left_shoulder" gear="10"/>
                <motor name="left_elbow_motor" joint="left_elbow" gear="10"/>
                <motor name="right_shoulder_motor" joint="right_shoulder" gear="10"/>
                <motor name="right_elbow_motor" joint="right_elbow" gear="10"/>
                <motor name="left_hip_motor" joint="left_hip" gear="20"/>
                <motor name="left_knee_motor" joint="left_knee" gear="20"/>
                <motor name="right_hip_motor" joint="right_hip" gear="20"/>
                <motor name="right_knee_motor" joint="right_knee" gear="20"/>
            </actuator>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.env = None
    
    def _setup_action_space(self):
        """Setup action space based on model."""
        if hasattr(self, 'env') and self.env is not None:
            self.action_space = self.env.action_space
        else:
            n_actions = self.model.nu
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(n_actions,),
                dtype=np.float32
            )
    
    def _setup_observation_space(self):
        """Setup observation space based on model."""
        if hasattr(self, 'env') and self.env is not None:
            self.observation_space = self.env.observation_space
        else:
            nq = self.model.nq
            nv = self.model.nv
            if self.exclude_current_positions_from_observation:
                obs_dim = nq - 2 + nv
            else:
                obs_dim = nq + nv
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from the environment."""
        if self.env is not None:
            return self.env.unwrapped._get_obs()
        else:
            qpos = self.data.qpos.flat.copy()
            qvel = self.data.qvel.flat.copy()
            if self.exclude_current_positions_from_observation:
                qpos = qpos[2:]
            return np.concatenate([qpos, qvel]).astype(np.float32)
    
    def _is_healthy(self) -> bool:
        """Check if the robot is in a healthy state."""
        if self.env is not None:
            z = self.env.unwrapped.data.qpos[2]
        else:
            z = self.data.qpos[2]
        return self.healthy_z_range[0] < z < self.healthy_z_range[1]
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (joint torques/positions)
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Additional information
        """
        if self.env is not None:
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            forward_vel = info.get('x_velocity', 0)
            current_x = self.env.unwrapped.data.qpos[0]
            
            forward_reward = self.forward_reward_weight * forward_vel
            
            backward_penalty = 0
            if forward_vel < -0.1:
                backward_penalty = self.backward_penalty_weight * abs(forward_vel)
            
            position_reward = 0
            if current_x > self.start_x_position:
                position_reward = self.position_reward_weight * (current_x - self.start_x_position)
            
            velocity_bonus = 0.0
            air_time_reward = 0.0
            energy_efficiency = 0.0
            fall_penalty = 0.0
            
            if self.task == "run":
                forward_reward *= 3.0
                position_reward *= 2.0
                
                if forward_vel > 2.0:
                    velocity_bonus = self.velocity_bonus_weight * (forward_vel - 2.0) ** 2
                
                foot_contact = self._check_foot_contact()
                is_healthy = self._is_healthy()
                
                if not foot_contact and self.last_foot_contact:
                    self.air_time = 0.0
                elif not foot_contact and is_healthy and forward_vel > 1.0:
                    self.air_time += 0.01
                    air_time_reward = self.air_time_reward_weight * self.air_time * min(forward_vel / 3.0, 1.0)
                elif not foot_contact and not is_healthy:
                    fall_penalty = 50.0
                    self.air_time = 0.0
                self.last_foot_contact = foot_contact
                
                if forward_vel > 0.5 and is_healthy:
                    energy_efficiency = self.energy_efficiency_weight * (forward_vel / (np.sum(np.abs(action)) + 1e-6))
            
            reward = reward + forward_reward + position_reward + velocity_bonus + air_time_reward + energy_efficiency - backward_penalty - fall_penalty
            
            info['task'] = self.task
            info['is_healthy'] = self._is_healthy()
            info['reward_forward'] = forward_reward
            info['reward_position'] = position_reward
            info['reward_velocity_bonus'] = velocity_bonus
            info['reward_air_time'] = air_time_reward
            info['reward_energy_efficiency'] = energy_efficiency
            info['penalty_backward'] = -backward_penalty
            info['penalty_fall'] = -fall_penalty
            info['x_position'] = current_x
            info['start_x_position'] = self.start_x_position
            info['distance_traveled'] = current_x - self.start_x_position
            info['air_time'] = self.air_time
            info['foot_contact'] = self.last_foot_contact
            
            if self.render_mode == "human":
                self.env.render()
            
            return observation, reward, terminated, truncated, info
        else:
            action = np.clip(action, -1.0, 1.0)
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)
            
            observation = self._get_observation()
            
            healthy = self._is_healthy()
            terminated = not healthy if self.terminate_when_unhealthy else False
            truncated = False
            
            forward_vel = self.data.qvel[0]
            current_x = self.data.qpos[0]
            
            forward_reward = self.forward_reward_weight * forward_vel
            
            backward_penalty = 0
            if forward_vel < -0.1:
                backward_penalty = self.backward_penalty_weight * abs(forward_vel)
            
            position_reward = 0
            if current_x > self.start_x_position:
                position_reward = self.position_reward_weight * (current_x - self.start_x_position)
            
            ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))
            
            contact_cost = 0
            if self.contact_cost_weight > 0:
                contact_forces = self.data.cfrc_ext
                contact_cost = self.contact_cost_weight * np.sum(np.square(contact_forces))
            
            healthy_reward = self.healthy_reward if healthy else 0
            
            velocity_bonus = 0.0
            air_time_reward = 0.0
            energy_efficiency = 0.0
            fall_penalty = 0.0
            
            if self.task == "run":
                forward_reward *= 3.0
                position_reward *= 2.0
                
                if forward_vel > 2.0:
                    velocity_bonus = self.velocity_bonus_weight * (forward_vel - 2.0) ** 2
                
                foot_contact = self._check_foot_contact()
                
                if not foot_contact and self.last_foot_contact:
                    self.air_time = 0.0
                elif not foot_contact and healthy and forward_vel > 1.0:
                    self.air_time += 0.01
                    air_time_reward = self.air_time_reward_weight * self.air_time * min(forward_vel / 3.0, 1.0)
                elif not foot_contact and not healthy:
                    fall_penalty = 50.0
                    self.air_time = 0.0
                self.last_foot_contact = foot_contact
                
                if forward_vel > 0.5 and healthy:
                    energy_efficiency = self.energy_efficiency_weight * (forward_vel / (np.sum(np.abs(action)) + 1e-6))
            
            reward = forward_reward + position_reward + healthy_reward + velocity_bonus + air_time_reward + energy_efficiency - ctrl_cost - contact_cost - backward_penalty - fall_penalty
            
            info = {
                'reward_forward': forward_reward,
                'reward_position': position_reward,
                'reward_velocity_bonus': velocity_bonus,
                'reward_air_time': air_time_reward,
                'reward_energy_efficiency': energy_efficiency,
                'reward_healthy': healthy_reward,
                'reward_ctrl': -ctrl_cost,
                'reward_contact': -contact_cost,
                'penalty_backward': -backward_penalty,
                'penalty_fall': -fall_penalty,
                'x_velocity': forward_vel,
                'x_position': current_x,
                'z_position': self.data.qpos[2],
                'start_x_position': self.start_x_position,
                'distance_traveled': current_x - self.start_x_position,
                'air_time': self.air_time,
                'foot_contact': self.last_foot_contact,
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
            observation: Initial observation
            info: Additional information
        """
        if self.env is not None:
            observation, info = self.env.reset(seed=seed, options=options)
            self.start_x_position = self.env.unwrapped.data.qpos[0]
            self.air_time = 0.0
            self.last_foot_contact = True
            info['task'] = self.task
            info['start_x_position'] = self.start_x_position
            info['air_time'] = self.air_time
            info['foot_contact'] = self.last_foot_contact
            return observation, info
        else:
            super().reset(seed=seed)
            
            mujoco.mj_resetData(self.model, self.data)
            
            qpos = self.data.qpos.flat.copy()
            qvel = self.data.qvel.flat.copy()
            
            noise_low = -self.reset_noise_scale
            noise_high = self.reset_noise_scale
            
            qpos += self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
            qvel += self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
            
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel
            
            mujoco.mj_forward(self.model, self.data)
            
            self.start_x_position = self.data.qpos[0]
            self.air_time = 0.0
            self.last_foot_contact = True
            
            observation = self._get_observation()
            info = {
                'task': self.task,
                'is_healthy': self._is_healthy(),
                'start_x_position': self.start_x_position,
                'air_time': self.air_time,
                'foot_contact': self.last_foot_contact
            }
            
            return observation, info
    
    def render(self):
        """Render the environment."""
        if self.env is not None:
            return self.env.render()
        
        if self.render_mode is None:
            return None
        
        if self.viewer is None:
            if self.render_mode == "human":
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer = mujoco.Renderer(self.model, height=480, width=640)
        
        if self.render_mode == "human":
            self.viewer.sync()
            return None
        else:
            self.viewer.update_scene(self.data)
            return self.viewer.render()
    
    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for logging/visualization."""
        if self.env is not None:
            return {
                'qpos': self.env.unwrapped.data.qpos.copy(),
                'qvel': self.env.unwrapped.data.qvel.copy(),
                'x_position': self.env.unwrapped.data.qpos[0],
                'y_position': self.env.unwrapped.data.qpos[1],
                'z_position': self.env.unwrapped.data.qpos[2],
                'x_velocity': self.env.unwrapped.data.qvel[0],
                'is_healthy': self._is_healthy()
            }
    
    def _check_foot_contact(self) -> bool:
        """Check if either foot is in contact with the ground."""
        try:
            if self.env is not None:
                data = self.env.unwrapped.data
            else:
                data = self.data
            
            for i in range(data.ncon):
                contact = data.contact[i]
                geom1_name = data.model.geom(contact.geom1).name
                geom2_name = data.model.geom(contact.geom2).name
                
                foot_geoms = ['foot', 'left_foot', 'right_foot', 'toe', 'heel']
                floor_geoms = ['floor', 'ground', 'plane']
                
                for foot in foot_geoms:
                    for floor in floor_geoms:
                        if (foot in geom1_name.lower() and floor in geom2_name.lower()) or \
                           (foot in geom2_name.lower() and floor in geom1_name.lower()):
                            return True
            
            return False
        except Exception as e:
            return True
