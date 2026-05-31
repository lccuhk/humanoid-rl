#!/usr/bin/env python3
"""
Optimized training script for humanoid running task.

This script uses an enhanced reward function to encourage
faster and more natural running gait.
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environments import HumanoidMuJoCoEnv
from src.agents import PPOAgent
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import csv


class DetailedLoggingCallback(BaseCallback):
    """Custom callback for detailed logging during training."""
    
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        self.csv_writer = None
        self.total_episodes = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _init_callback(self):
        """Initialize callback."""
        if self.csv_writer is None:
            self.csv_file = open(self.log_file, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'step', 'episode', 'reward', 'reward_forward', 'reward_position',
                'reward_velocity_bonus', 'reward_air_time', 'reward_energy_efficiency',
                'reward_healthy', 'reward_ctrl', 'reward_contact', 'penalty_backward',
                'penalty_fall', 'x_velocity', 'x_position', 'z_position', 'distance_traveled',
                'air_time', 'foot_contact', 'timestamp'
            ])
    
    def _on_step(self):
        """Called after each step."""
        reward = self.locals.get('rewards', [0])[0]
        info = self.locals.get('infos', [{}])[0]
        done = self.locals.get('dones', [False])[0]
        
        self.current_episode_reward += float(reward)
        
        step_log = {
            'step': self.num_timesteps,
            'episode': self.total_episodes + 1,
            'reward': float(reward),
            'reward_forward': float(info.get('reward_forward', 0)),
            'reward_position': float(info.get('reward_position', 0)),
            'reward_velocity_bonus': float(info.get('reward_velocity_bonus', 0)),
            'reward_air_time': float(info.get('reward_air_time', 0)),
            'reward_energy_efficiency': float(info.get('reward_energy_efficiency', 0)),
            'reward_healthy': float(info.get('reward_healthy', 0)),
            'reward_ctrl': float(info.get('reward_ctrl', 0)),
            'reward_contact': float(info.get('reward_contact', 0)),
            'penalty_backward': float(info.get('penalty_backward', 0)),
            'penalty_fall': float(info.get('penalty_fall', 0)),
            'x_velocity': float(info.get('x_velocity', 0)),
            'x_position': float(info.get('x_position', 0)),
            'z_position': float(info.get('z_position', 0)),
            'distance_traveled': float(info.get('distance_traveled', 0)),
            'air_time': float(info.get('air_time', 0)),
            'foot_contact': int(info.get('foot_contact', True)),
            'timestamp': datetime.now().isoformat()
        }
        
        self.csv_writer.writerow([
            step_log['step'], step_log['episode'], step_log['reward'],
            step_log['reward_forward'], step_log['reward_position'],
            step_log['reward_velocity_bonus'], step_log['reward_air_time'],
            step_log['reward_energy_efficiency'], step_log['reward_healthy'],
            step_log['reward_ctrl'], step_log['reward_contact'],
            step_log['penalty_backward'], step_log['penalty_fall'],
            step_log['x_velocity'], step_log['x_position'], step_log['z_position'],
            step_log['distance_traveled'], step_log['air_time'],
            step_log['foot_contact'], step_log['timestamp']
        ])
        
        if done:
            self.total_episodes += 1
            self.episode_rewards.append(self.current_episode_reward)
            
            if self.total_episodes % 100 == 0:
                avg_reward = sum(self.episode_rewards[-100:]) / min(100, len(self.episode_rewards))
                print(f"Episode {self.total_episodes} | Avg Reward (last 100): {avg_reward:.2f} | "
                      f"Steps: {self.num_timesteps}")
            
            self.current_episode_reward = 0
        
        return True
    
    def _on_training_end(self):
        """Called at the end of training."""
        if self.csv_file:
            self.csv_file.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train optimized humanoid running policy'
    )
    
    parser.add_argument(
        '--total_timesteps', type=int, default=2000000,
        help='Total training timesteps (default: 2,000,000)'
    )
    
    parser.add_argument(
        '--learning_rate', type=float, default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    
    parser.add_argument(
        '--n_steps', type=int, default=2048,
        help='Number of steps per update (default: 2048)'
    )
    
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size (default: 64)'
    )
    
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='Discount factor (default: 0.99)'
    )
    
    parser.add_argument(
        '--gae_lambda', type=float, default=0.95,
        help='GAE lambda (default: 0.95)'
    )
    
    parser.add_argument(
        '--clip_range', type=float, default=0.2,
        help='PPO clip range (default: 0.2)'
    )
    
    parser.add_argument(
        '--velocity_bonus_weight', type=float, default=30.0,
        help='Weight for velocity bonus reward (default: 30.0)'
    )
    
    parser.add_argument(
        '--air_time_reward_weight', type=float, default=50.0,
        help='Weight for air time reward (default: 50.0)'
    )
    
    parser.add_argument(
        '--energy_efficiency_weight', type=float, default=5.0,
        help='Weight for energy efficiency reward (default: 5.0)'
    )
    
    parser.add_argument(
        '--checkpoint_dir', type=str, default='./checkpoints',
        help='Directory to save checkpoints (default: ./checkpoints)'
    )
    
    parser.add_argument(
        '--log_dir', type=str, default='./logs',
        help='Directory to save logs (default: ./logs)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_mujoco_run_optimized_{timestamp}"
    
    print("=" * 70)
    print("Optimized Humanoid Running Training")
    print("=" * 70)
    print(f"Run name: {run_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Velocity bonus weight: {args.velocity_bonus_weight}")
    print(f"Air time reward weight: {args.air_time_reward_weight}")
    print(f"Energy efficiency weight: {args.energy_efficiency_weight}")
    print("=" * 70)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("\nCreating environment...")
    env = HumanoidMuJoCoEnv(
        render_mode=None,
        task='run',
        forward_reward_weight=5.0,
        healthy_reward=1.0,
        backward_penalty_weight=10.0,
        position_reward_weight=2.0,
        velocity_bonus_weight=args.velocity_bonus_weight,
        air_time_reward_weight=args.air_time_reward_weight,
        energy_efficiency_weight=args.energy_efficiency_weight,
    )
    
    print("Creating agent...")
    agent = PPOAgent(
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        verbose=1,
        tensorboard_log=os.path.join(args.log_dir, 'tensorboard', run_name),
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=args.checkpoint_dir,
        name_prefix=run_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    logging_callback = DetailedLoggingCallback(
        log_dir=os.path.join(args.log_dir, 'training', run_name),
        verbose=1,
    )
    
    print("\nStarting training...")
    print("=" * 70)
    
    try:
        agent.train(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, logging_callback],
            tb_log_name=run_name,
            reset_num_timesteps=True,
            progress_bar=True,
        )
        
        final_model_path = os.path.join(args.checkpoint_dir, f"{run_name}_final.zip")
        agent.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        final_model_path = os.path.join(args.checkpoint_dir, f"{run_name}_interrupted.zip")
        agent.save(final_model_path)
        print(f"Model saved to: {final_model_path}")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    main()
