#!/usr/bin/env python3
"""
Training script for Humanoid Robot Reinforcement Learning.

This script trains a humanoid robot to walk and run using
deep reinforcement learning with multiple physics engines
and algorithms.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src.environments import HumanoidMuJoCoEnv, HumanoidPyBulletEnv
from src.agents import PPOAgent, SACAgent, TD3Agent
from src.data_processing import DataCollector, DataPreprocessor
from src.visualization import TrainingVisualizer, RobotVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train humanoid robot with deep reinforcement learning'
    )
    
    parser.add_argument(
        '--env', type=str, default='mujoco',
        choices=['mujoco', 'pybullet', 'isaacgym'],
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
        '--save_interval', type=int, default=100000,
        help='Save model every N timesteps (default: 100000)'
    )
    
    parser.add_argument(
        '--log_dir', type=str, default='./logs',
        help='Directory for logs (default: ./logs)'
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
    elif args.env == 'isaacgym':
        print(f"Creating Isaac Gym environment for {args.task} task...")
        from src.environments import HumanoidIsaacGymEnv
        env = HumanoidIsaacGymEnv(
            render_mode=render_mode,
            task=args.task,
        )
    else:
        raise ValueError(f"Unknown environment: {args.env}")
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    return env


def create_agent(args, env):
    """Create the RL agent."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log = os.path.join(args.log_dir, f"tb_{args.algo}_{args.env}_{args.task}_{timestamp}")
    
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
    """Main training function."""
    print("="*70)
    print("Humanoid Robot Reinforcement Learning - Training")
    print("="*70)
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")
    print(f"Task: {args.task}")
    print(f"Total Timesteps: {args.total_timesteps}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Gamma: {args.gamma}")
    print(f"Seed: {args.seed}")
    print("="*70)
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    env = create_environment(args)
    
    agent = create_agent(args, env)
    
    data_collector = DataCollector(
        output_dir=os.path.join(args.log_dir, 'data'),
        save_interval=100,
    )
    
    training_visualizer = TrainingVisualizer(
        output_dir=os.path.join(args.log_dir, 'visualizations'),
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.algo}_{args.env}_{args.task}_{timestamp}"
    
    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    print(f"Model name: {model_name}")
    print("-"*70)
    
    try:
        agent.train(
            total_timesteps=args.total_timesteps,
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
        
        print("\nGenerating visualizations...")
        training_visualizer.create_training_dashboard(save=True)
        
        training_config = {
            'environment': args.env,
            'algorithm': args.algo,
            'task': args.task,
            'total_timesteps': args.total_timesteps,
            'learning_rate': args.learning_rate,
            'gamma': args.gamma,
            'seed': args.seed,
            'timestamp': timestamp,
            'model_path': final_checkpoint_path,
            'evaluation_results': eval_results,
        }
        
        config_path = os.path.join(args.log_dir, f"config_{model_name}.json")
        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        print(f"Training config saved to: {config_path}")
        
        data_collector.save_metrics()
        
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


if __name__ == "__main__":
    args = parse_args()
    train(args)
