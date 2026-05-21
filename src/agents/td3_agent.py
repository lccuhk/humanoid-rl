"""TD3 Agent for Humanoid Robot Locomotion"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


class TD3Agent:
    """
    Twin Delayed DDPG (TD3) agent for humanoid robot locomotion.
    
    TD3 is an improved version of DDPG that addresses function approximation
    error issues through:
    1. Clipped Double Q-learning
    2. Delayed Policy Updates
    3. Target Policy Smoothing
    """
    
    def __init__(
        self,
        env,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Tuple[int, str] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[str] = "normal",
        replay_buffer_class: Optional[Any] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        device: str = "auto",
        verbose: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Initialize TD3 agent.
        
        Args:
            env: Gymnasium environment
            policy: Policy type
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            learning_starts: How many steps of the model to collect transitions for before learning starts
            batch_size: Minibatch size
            tau: Target smoothing coefficient
            gamma: Discount factor
            train_freq: Update the model every train_freq steps
            gradient_steps: How many gradient steps to do after each rollout
            action_noise: Type of action noise ('normal', 'ornstein-uhlenbeck', or None)
            replay_buffer_class: Replay buffer class
            replay_buffer_kwargs: Replay buffer kwargs
            optimize_memory_usage: Enable a memory efficient variant
            policy_delay: Policy and target networks will only be updated once every policy_delay steps
            target_policy_noise: Standard deviation of Gaussian noise added to target policy
            target_noise_clip: Limit for absolute value of target policy smoothing noise
            tensorboard_log: Tensorboard log location
            device: Device to use
            verbose: Verbosity level
            seed: Random seed
        """
        self.env = env
        self.policy = policy
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.action_noise_type = action_noise
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs
        self.optimize_memory_usage = optimize_memory_usage
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.tensorboard_log = tensorboard_log
        self.device = device
        self.verbose = verbose
        self.seed = seed
        
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build the TD3 model."""
        action_dim = self.env.action_space.shape[-1]
        
        if self.action_noise_type == "normal":
            action_noise = NormalActionNoise(
                mean=np.zeros(action_dim),
                sigma=0.1 * np.ones(action_dim)
            )
        elif self.action_noise_type == "ornstein-uhlenbeck":
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(action_dim),
                sigma=0.2 * np.ones(action_dim)
            )
        else:
            action_noise = None
        
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[400, 300], qf=[400, 300])
        )
        
        self.model = TD3(
            policy=self.policy,
            env=self.env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=self.replay_buffer_class,
            replay_buffer_kwargs=self.replay_buffer_kwargs,
            optimize_memory_usage=self.optimize_memory_usage,
            policy_delay=self.policy_delay,
            target_policy_noise=self.target_policy_noise,
            target_noise_clip=self.target_noise_clip,
            tensorboard_log=self.tensorboard_log,
            policy_kwargs=policy_kwargs,
            device=self.device,
            verbose=self.verbose,
            seed=self.seed,
        )
    
    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Train the agent.
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Callback function
            log_interval: Log interval
            tb_log_name: Tensorboard log name
            reset_num_timesteps: Whether to reset number of timesteps
            progress_bar: Whether to show progress bar
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]:
        """
        Predict action from observation.
        
        Args:
            observation: Environment observation
            state: Last hidden state
            episode_start: Whether the episode started a new episode
            deterministic: Whether to return deterministic actions
            
        Returns:
            action: Predicted action
            state: New hidden state
        """
        return self.model.predict(
            observation=observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
    
    def save(self, path: str):
        """
        Save the model to file.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
    
    def load(self, path: str, env=None):
        """
        Load model from file.
        
        Args:
            path: Path to load the model
            env: Environment to load the model into
        """
        self.model = TD3.load(path, env=env)
    
    def evaluate(self, env, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate the agent.
        
        Args:
            env: Evaluation environment
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary with evaluation metrics
        """
        rewards = []
        episode_lengths = []
        forward_distances = []
        max_velocities = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            max_velocity = 0
            start_pos = None
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if 'x_velocity' in info:
                    max_velocity = max(max_velocity, info['x_velocity'])
                
                if start_pos is None and 'x_position' in info:
                    start_pos = info['x_position']
                
                done = terminated or truncated
            
            rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            max_velocities.append(max_velocity)
            
            if start_pos is not None and 'x_position' in info:
                forward_distances.append(info['x_position'] - start_pos)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_forward_distance': np.mean(forward_distances) if forward_distances else 0,
            'mean_max_velocity': np.mean(max_velocities),
        }
