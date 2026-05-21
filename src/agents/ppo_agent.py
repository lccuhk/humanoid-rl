"""PPO Agent for Humanoid Robot Locomotion"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, Any, Optional, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for humanoid robot locomotion.
    
    This agent uses Stable Baselines3's PPO implementation for training
    humanoid robots to walk and run.
    """
    
    def __init__(
        self,
        env,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        device: str = "auto",
        verbose: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Initialize PPO agent.
        
        Args:
            env: Gymnasium environment
            policy: Policy type ('MlpPolicy' or 'CnnPolicy')
            learning_rate: Learning rate
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epochs when optimizing the surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            clip_range: Clipping parameter
            clip_range_vf: Clipping parameter for the value function
            ent_coef: Entropy coefficient for the loss calculation
            vf_coef: Value function coefficient for the loss calculation
            max_grad_norm: The maximum value for the gradient clipping
            use_sde: Whether to use generalized State Dependent Exploration
            sde_sample_freq: Sample a new noise matrix every n steps
            target_kl: Limit the KL divergence between updates
            tensorboard_log: Tensorboard log location
            device: Device to use ('auto', 'cuda', 'cpu')
            verbose: Verbosity level
            seed: Random seed
        """
        self.env = env
        self.policy = policy
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.target_kl = target_kl
        self.tensorboard_log = tensorboard_log
        self.device = device
        self.verbose = verbose
        self.seed = seed
        
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build the PPO model."""
        policy_kwargs = dict(
            activation_fn=nn.Tanh,
            net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])
        )
        
        self.model = PPO(
            policy=self.policy,
            env=self.env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            clip_range_vf=self.clip_range_vf,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            use_sde=self.use_sde,
            sde_sample_freq=self.sde_sample_freq,
            target_kl=self.target_kl,
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
        log_interval: int = 1,
        tb_log_name: str = "PPO",
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
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Predict action from observation.
        
        Args:
            observation: Environment observation
            state: Last hidden state (for recurrent policies)
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
        self.model = PPO.load(path, env=env)
    
    def get_action_probabilities(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Get action probabilities for an observation.
        
        Args:
            observation: Environment observation
            
        Returns:
            Dictionary containing action probabilities and log probabilities
        """
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_dist = self.model.policy.get_distribution(obs_tensor)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        return {
            'action': action.cpu().numpy().squeeze(),
            'log_prob': log_prob.cpu().numpy().squeeze(),
            'mean': action_dist.distribution.mean.cpu().numpy().squeeze(),
            'std': action_dist.distribution.stddev.cpu().numpy().squeeze()
        }
    
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
