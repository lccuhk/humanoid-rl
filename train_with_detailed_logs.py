#!/usr/bin/env python3
"""
Training script with detailed logging for Humanoid Robot RL.

This script provides detailed logging of every step's reward changes,
including:
- Step-by-step reward breakdown
- Episode-level statistics
- Real-time progress monitoring
- CSV and JSON log files
"""

import os
import sys
import argparse
import json
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.environments import HumanoidMuJoCoEnv, HumanoidPyBulletEnv
from src.agents import PPOAgent, SACAgent, TD3Agent


class DetailedLoggingCallback(BaseCallback):
    """
    Custom callback for detailed logging during training.
    
    Records:
    - Step-by-step rewards
    - Episode statistics
    - Reward component breakdown
    - Environment state information
    """
    
    def __init__(
        self,
        log_dir: str,
        log_interval: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        
        self.log_dir = log_dir
        self.log_interval = log_interval
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.step_logs = []
        self.episode_logs = []
        
        self.current_episode_rewards = []
        self.current_episode_steps = 0
        self.current_episode_start = datetime.now()
        
        self.total_steps = 0
        self.total_episodes = 0
        
        self.csv_file = None
        self.csv_writer = None
        self._init_csv_logger()
        
        self.start_time = datetime.now()
    
    def _init_csv_logger(self):
        """Initialize CSV logger."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.log_dir, f"step_logs_{timestamp}.csv")
        
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        headers = [
            'step',
            'episode',
            'reward',
            'reward_forward',
            'reward_position',
            'reward_healthy',
            'reward_ctrl',
            'reward_contact',
            'penalty_backward',
            'x_velocity',
            'x_position',
            'z_position',
            'distance_traveled',
            'is_healthy',
            'terminated',
            'truncated',
            'timestamp'
        ]
        self.csv_writer.writerow(headers)
    
    def _on_step(self) -> bool:
        """
        Called after each step in the environment.
        
        Returns:
            bool: Whether training should continue
        """
        self.total_steps += 1
        self.current_episode_steps += 1
        
        infos = self.locals.get('infos', [{}])
        
        for info in infos:
            reward = self.locals.get('rewards', [0])[0]
            
            step_log = {
                'step': self.total_steps,
                'episode': self.total_episodes + 1,
                'reward': float(reward),
                'reward_forward': float(info.get('reward_forward', 0)),
                'reward_position': float(info.get('reward_position', 0)),
                'reward_healthy': float(info.get('reward_healthy', 0)),
                'reward_ctrl': float(info.get('reward_ctrl', 0)),
                'reward_contact': float(info.get('reward_contact', 0)),
                'penalty_backward': float(info.get('penalty_backward', 0)),
                'x_velocity': float(info.get('x_velocity', 0)),
                'x_position': float(info.get('x_position', 0)),
                'z_position': float(info.get('z_position', 0)),
                'distance_traveled': float(info.get('distance_traveled', 0)),
                'is_healthy': bool(info.get('is_healthy', True)),
                'terminated': bool(self.locals.get('dones', [False])[0]),
                'truncated': False,
                'timestamp': datetime.now().isoformat()
            }
            
            self.step_logs.append(step_log)
            self.current_episode_rewards.append(float(reward))
            
            self.csv_writer.writerow([
                step_log['step'],
                step_log['episode'],
                step_log['reward'],
                step_log['reward_forward'],
                step_log['reward_position'],
                step_log['reward_healthy'],
                step_log['reward_ctrl'],
                step_log['reward_contact'],
                step_log['penalty_backward'],
                step_log['x_velocity'],
                step_log['x_position'],
                step_log['z_position'],
                step_log['distance_traveled'],
                step_log['is_healthy'],
                step_log['terminated'],
                step_log['truncated'],
                step_log['timestamp']
            ])
            
            if self.locals.get('dones', [False])[0]:
                self._on_episode_end()
        
        if self.total_steps % self.log_interval == 0:
            self._log_progress()
        
        return True
    
    def _on_episode_end(self):
        """Called at the end of an episode."""
        self.total_episodes += 1
        
        episode_duration = (datetime.now() - self.current_episode_start).total_seconds()
        
        episode_log = {
            'episode': self.total_episodes,
            'total_steps': self.total_steps,
            'episode_steps': self.current_episode_steps,
            'total_reward': sum(self.current_episode_rewards),
            'mean_reward_per_step': np.mean(self.current_episode_rewards),
            'min_reward': min(self.current_episode_rewards),
            'max_reward': max(self.current_episode_rewards),
            'std_reward': np.std(self.current_episode_rewards),
            'duration_seconds': episode_duration,
            'steps_per_second': self.current_episode_steps / episode_duration if episode_duration > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.episode_logs.append(episode_log)
        
        print(f"\n{'='*70}")
        print(f"Episode {self.total_episodes} Completed")
        print(f"{'='*70}")
        print(f"  Total Steps: {self.total_steps:,}")
        print(f"  Episode Steps: {self.current_episode_steps}")
        print(f"  Total Reward: {episode_log['total_reward']:.2f}")
        print(f"  Mean Reward/Step: {episode_log['mean_reward_per_step']:.4f}")
        print(f"  Reward Range: [{episode_log['min_reward']:.2f}, {episode_log['max_reward']:.2f}]")
        print(f"  Duration: {episode_duration:.2f}s ({episode_log['steps_per_second']:.1f} steps/s)")
        print(f"{'='*70}\n")
        
        self.current_episode_rewards = []
        self.current_episode_steps = 0
        self.current_episode_start = datetime.now()
    
    def _log_progress(self):
        """Log training progress."""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\n{'='*70}")
        print(f"Training Progress - Step {self.total_steps:,}")
        print(f"{'='*70}")
        print(f"  Total Steps: {self.total_steps:,}")
        print(f"  Total Episodes: {self.total_episodes}")
        print(f"  Elapsed Time: {elapsed_time:.1f}s")
        print(f"  Steps/Second: {self.total_steps / elapsed_time:.1f}")
        
        if self.step_logs:
            recent_rewards = [log['reward'] for log in self.step_logs[-100:]]
            print(f"  Recent Mean Reward: {np.mean(recent_rewards):.4f}")
            print(f"  Recent Reward Range: [{min(recent_rewards):.4f}, {max(recent_rewards):.4f}]")
        
        if self.episode_logs:
            recent_episodes = self.episode_logs[-10:]
            print(f"  Recent Mean Episode Reward: {np.mean([e['total_reward'] for e in recent_episodes]):.2f}")
            print(f"  Recent Mean Episode Length: {np.mean([e['episode_steps'] for e in recent_episodes]):.1f}")
        
        print(f"{'='*70}\n")
        
        self._save_intermediate_logs()
    
    def _save_intermediate_logs(self):
        """Save intermediate logs to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        step_logs_path = os.path.join(self.log_dir, f"step_logs_intermediate_{timestamp}.json")
        with open(step_logs_path, 'w') as f:
            json.dump(self.step_logs[-1000:], f, indent=2)
        
        episode_logs_path = os.path.join(self.log_dir, f"episode_logs_{timestamp}.json")
        with open(episode_logs_path, 'w') as f:
            json.dump(self.episode_logs, f, indent=2)
    
    def _on_training_end(self):
        """Called at the end of training."""
        print(f"\n{'='*70}")
        print("Training Completed - Saving Final Logs")
        print(f"{'='*70}")
        
        self.csv_file.close()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        step_logs_path = os.path.join(self.log_dir, f"step_logs_final_{timestamp}.json")
        with open(step_logs_path, 'w') as f:
            json.dump(self.step_logs, f, indent=2)
        print(f"Step logs saved to: {step_logs_path}")
        
        episode_logs_path = os.path.join(self.log_dir, f"episode_logs_final_{timestamp}.json")
        with open(episode_logs_path, 'w') as f:
            json.dump(self.episode_logs, f, indent=2)
        print(f"Episode logs saved to: {episode_logs_path}")
        
        summary = self._generate_summary()
        summary_path = os.path.join(self.log_dir, f"training_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Training summary saved to: {summary_path}")
        
        self._print_summary(summary)
        
        print(f"{'='*70}\n")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate training summary statistics."""
        if not self.step_logs:
            return {}
        
        rewards = [log['reward'] for log in self.step_logs]
        forward_rewards = [log['reward_forward'] for log in self.step_logs]
        position_rewards = [log['reward_position'] for log in self.step_logs]
        healthy_rewards = [log['reward_healthy'] for log in self.step_logs]
        ctrl_costs = [log['reward_ctrl'] for log in self.step_logs]
        backward_penalties = [log['penalty_backward'] for log in self.step_logs]
        x_velocities = [log['x_velocity'] for log in self.step_logs]
        distances = [log['distance_traveled'] for log in self.step_logs]
        
        episode_rewards = [e['total_reward'] for e in self.episode_logs]
        episode_lengths = [e['episode_steps'] for e in self.episode_logs]
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'metadata': {
                'total_steps': self.total_steps,
                'total_episodes': self.total_episodes,
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': total_time,
                'steps_per_second': self.total_steps / total_time if total_time > 0 else 0,
            },
            'step_statistics': {
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'min_reward': float(np.min(rewards)),
                'max_reward': float(np.max(rewards)),
                'mean_forward_reward': float(np.mean(forward_rewards)),
                'mean_position_reward': float(np.mean(position_rewards)),
                'mean_healthy_reward': float(np.mean(healthy_rewards)),
                'mean_ctrl_cost': float(np.mean(ctrl_costs)),
                'mean_backward_penalty': float(np.mean(backward_penalties)),
                'mean_x_velocity': float(np.mean(x_velocities)),
                'max_x_velocity': float(np.max(x_velocities)),
                'mean_distance_traveled': float(np.mean(distances)),
                'max_distance_traveled': float(np.max(distances)),
            },
            'episode_statistics': {
                'mean_episode_reward': float(np.mean(episode_rewards)),
                'std_episode_reward': float(np.std(episode_rewards)),
                'min_episode_reward': float(np.min(episode_rewards)),
                'max_episode_reward': float(np.max(episode_rewards)),
                'mean_episode_length': float(np.mean(episode_lengths)),
                'std_episode_length': float(np.std(episode_lengths)),
                'min_episode_length': float(np.min(episode_lengths)),
                'max_episode_length': float(np.max(episode_lengths)),
            },
            'reward_breakdown': {
                'forward_reward_ratio': float(np.sum(forward_rewards) / np.sum(rewards)) if np.sum(rewards) != 0 else 0,
                'position_reward_ratio': float(np.sum(position_rewards) / np.sum(rewards)) if np.sum(rewards) != 0 else 0,
                'healthy_reward_ratio': float(np.sum(healthy_rewards) / np.sum(rewards)) if np.sum(rewards) != 0 else 0,
                'ctrl_cost_ratio': float(np.sum(ctrl_costs) / np.sum(rewards)) if np.sum(rewards) != 0 else 0,
                'backward_penalty_ratio': float(np.sum(backward_penalties) / np.sum(rewards)) if np.sum(rewards) != 0 else 0,
            }
        }
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print training summary."""
        print(f"\n{'='*70}")
        print("Training Summary")
        print(f"{'='*70}")
        
        meta = summary.get('metadata', {})
        print(f"\nMetadata:")
        print(f"  Total Steps: {meta.get('total_steps', 0):,}")
        print(f"  Total Episodes: {meta.get('total_episodes', 0)}")
        print(f"  Total Duration: {meta.get('total_duration_seconds', 0):.1f}s")
        print(f"  Steps/Second: {meta.get('steps_per_second', 0):.1f}")
        
        step_stats = summary.get('step_statistics', {})
        print(f"\nStep Statistics:")
        print(f"  Mean Reward: {step_stats.get('mean_reward', 0):.4f}")
        print(f"  Reward Range: [{step_stats.get('min_reward', 0):.4f}, {step_stats.get('max_reward', 0):.4f}]")
        print(f"  Mean Forward Velocity: {step_stats.get('mean_x_velocity', 0):.4f} m/s")
        print(f"  Max Forward Velocity: {step_stats.get('max_x_velocity', 0):.4f} m/s")
        print(f"  Mean Distance Traveled: {step_stats.get('mean_distance_traveled', 0):.4f} m")
        
        ep_stats = summary.get('episode_statistics', {})
        print(f"\nEpisode Statistics:")
        print(f"  Mean Episode Reward: {ep_stats.get('mean_episode_reward', 0):.2f}")
        print(f"  Best Episode Reward: {ep_stats.get('max_episode_reward', 0):.2f}")
        print(f"  Mean Episode Length: {ep_stats.get('mean_episode_length', 0):.1f} steps")
        print(f"  Longest Episode: {ep_stats.get('max_episode_length', 0):.1f} steps")
        
        breakdown = summary.get('reward_breakdown', {})
        print(f"\nReward Breakdown:")
        print(f"  Forward Reward: {breakdown.get('forward_reward_ratio', 0)*100:.1f}%")
        print(f"  Position Reward: {breakdown.get('position_reward_ratio', 0)*100:.1f}%")
        print(f"  Healthy Reward: {breakdown.get('healthy_reward_ratio', 0)*100:.1f}%")
        print(f"  Control Cost: {breakdown.get('ctrl_cost_ratio', 0)*100:.1f}%")
        print(f"  Backward Penalty: {breakdown.get('backward_penalty_ratio', 0)*100:.1f}%")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train humanoid robot with detailed logging'
    )
    
    parser.add_argument(
        '--env', type=str, default='mujoco',
        choices=['mujoco', 'pybullet'],
        help='Physics engine to use (default: mujoco)'
    )
    
    parser.add_argument(
        '--algo', type=str, default='ppo',
        choices=['ppo', 'sac', 'td3'],
        help='RL algorithm to use (default: ppo)'
    )
    
    parser.add_argument(
        '--task', type=str, default='walk',
        choices=['walk', 'run'],
        help='Task to train (default: walk)'
    )
    
    parser.add_argument(
        '--total_timesteps', type=int, default=1000000,
        help='Total training timesteps (default: 1000000)'
    )
    
    parser.add_argument(
        '--learning_rate', type=float, default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='Discount factor (default: 0.99)'
    )
    
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--log_interval', type=int, default=1000,
        help='Log progress every N steps (default: 1000)'
    )
    
    parser.add_argument(
        '--log_dir', type=str, default='./logs/detailed',
        help='Directory for detailed logs (default: ./logs/detailed)'
    )
    
    parser.add_argument(
        '--checkpoint_dir', type=str, default='./checkpoints',
        help='Directory for checkpoints (default: ./checkpoints)'
    )
    
    parser.add_argument(
        '--render', action='store_true',
        help='Render environment during training'
    )
    
    parser.add_argument(
        '--verbose', type=int, default=1,
        help='Verbosity level (default: 1)'
    )
    
    return parser.parse_args()


def create_environment(args):
    """Create the training environment."""
    render_mode = 'human' if args.render else None
    
    if args.env == 'mujoco':
        print(f"Creating MuJoCo environment for {args.task} task...")
        env = HumanoidMuJoCoEnv(
            render_mode=render_mode,
            task=args.task,
        )
    elif args.env == 'pybullet':
        print(f"Creating PyBullet environment for {args.task} task...")
        env = HumanoidPyBulletEnv(
            render_mode=render_mode,
            task=args.task,
        )
    else:
        raise ValueError(f"Unknown environment: {args.env}")
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    return env


def create_agent(args, env, log_dir: str):
    """Create the RL agent."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log = os.path.join(log_dir, f"tb_{args.algo}_{args.env}_{args.task}_{timestamp}")
    
    if args.algo == 'ppo':
        print(f"Creating PPO agent...")
        agent = PPOAgent(
            env=env,
            policy='MlpPolicy',
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            tensorboard_log=tensorboard_log,
            verbose=args.verbose,
            seed=args.seed,
        )
    elif args.algo == 'sac':
        print(f"Creating SAC agent...")
        agent = SACAgent(
            env=env,
            policy='MlpPolicy',
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            tensorboard_log=tensorboard_log,
            verbose=args.verbose,
            seed=args.seed,
        )
    elif args.algo == 'td3':
        print(f"Creating TD3 agent...")
        agent = TD3Agent(
            env=env,
            policy='MlpPolicy',
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            tensorboard_log=tensorboard_log,
            verbose=args.verbose,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")
    
    return agent


def train(args):
    """Main training function with detailed logging."""
    print("="*70)
    print("Humanoid Robot RL Training - Detailed Logging Mode")
    print("="*70)
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")
    print(f"Task: {args.task}")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Gamma: {args.gamma}")
    print(f"Seed: {args.seed}")
    print(f"Log Interval: {args.log_interval} steps")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.algo}_{args.env}_{args.task}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"\nLog Directory: {log_dir}")
    
    env = create_environment(args)
    
    agent = create_agent(args, env, log_dir)
    
    callback = DetailedLoggingCallback(
        log_dir=log_dir,
        log_interval=args.log_interval,
        verbose=args.verbose,
    )
    
    model_name = f"{args.algo}_{args.env}_{args.task}_{timestamp}"
    
    print(f"\nStarting training for {args.total_timesteps:,} timesteps...")
    print(f"Model name: {model_name}")
    print("-"*70)
    
    try:
        agent.train(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=1,
            tb_log_name=model_name,
            progress_bar=True,
        )
        
        print("\nTraining completed!")
        
        final_checkpoint_path = os.path.join(args.checkpoint_dir, f"{model_name}_final.zip")
        agent.save(final_checkpoint_path)
        print(f"Final model saved to: {final_checkpoint_path}")
        
        print("\nEvaluating trained model...")
        eval_results = agent.evaluate(env, n_episodes=10, deterministic=True)
        
        print("\nEvaluation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        training_config = {
            'environment': args.env,
            'algorithm': args.algo,
            'task': args.task,
            'total_timesteps': args.total_timesteps,
            'learning_rate': args.learning_rate,
            'gamma': args.gamma,
            'seed': args.seed,
            'log_interval': args.log_interval,
            'timestamp': timestamp,
            'model_path': final_checkpoint_path,
            'log_dir': log_dir,
            'evaluation_results': eval_results,
        }
        
        config_path = os.path.join(log_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        print(f"Training config saved to: {config_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{model_name}_interrupted.zip")
        agent.save(checkpoint_path)
        print(f"Model saved to: {checkpoint_path}")
    
    finally:
        env.close()
        print("\nEnvironment closed.")
    
    print("\n" + "="*70)
    print("Training session completed!")
    print("="*70)
    print(f"\nDetailed logs saved to: {log_dir}")
    print("Files generated:")
    print("  - step_logs_*.csv: Step-by-step reward breakdown")
    print("  - step_logs_*.json: Detailed step logs (JSON format)")
    print("  - episode_logs_*.json: Episode-level statistics")
    print("  - training_summary_*.json: Complete training summary")
    print("  - training_config.json: Training configuration")


if __name__ == "__main__":
    args = parse_args()
    train(args)
