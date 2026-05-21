#!/usr/bin/env python3
"""
Quick PPO Test - Simplified Environment

This script provides a quick way to verify that the PPO algorithm
is working correctly using lightweight Gymnasium environments.

Environments available:
- CartPole-v1: Discrete action space, very fast
- Pendulum-v1: Continuous action space, similar to humanoid
- MountainCarContinuous-v0: Continuous action space
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Quick PPO test with simple environments'
    )
    
    parser.add_argument(
        '--env', type=str, default='CartPole-v1',
        choices=['CartPole-v1', 'Pendulum-v1', 'MountainCarContinuous-v0', 'LunarLander-v2'],
        help='Environment to test (default: CartPole-v1)'
    )
    
    parser.add_argument(
        '--total_timesteps', type=int, default=50000,
        help='Total training timesteps (default: 50000)'
    )
    
    parser.add_argument(
        '--learning_rate', type=float, default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    
    parser.add_argument(
        '--n_steps', type=int, default=2048,
        help='Steps per update (default: 2048)'
    )
    
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Mini-batch size (default: 64)'
    )
    
    parser.add_argument(
        '--n_epochs', type=int, default=10,
        help='Epochs per update (default: 10)'
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
        '--eval_episodes', type=int, default=5,
        help='Number of evaluation episodes (default: 5)'
    )
    
    parser.add_argument(
        '--render', action='store_true',
        help='Render evaluation episodes'
    )
    
    parser.add_argument(
        '--save_model', action='store_true',
        help='Save trained model'
    )
    
    parser.add_argument(
        '--output_dir', type=str, default='./logs/quick_test',
        help='Output directory (default: ./logs/quick_test)'
    )
    
    return parser.parse_args()


def get_env_info(env_name: str) -> dict:
    """Get information about the environment."""
    env_info = {
        'CartPole-v1': {
            'description': 'Classic control task - balance a pole on a cart',
            'action_space': 'Discrete (2 actions)',
            'observation_space': 'Box (4 dimensions)',
            'difficulty': 'Easy',
            'expected_reward': '500 (max)',
            'training_time': '~1-2 minutes'
        },
        'Pendulum-v1': {
            'description': 'Classic control task - swing up and balance a pendulum',
            'action_space': 'Continuous (1 dimension)',
            'observation_space': 'Box (3 dimensions)',
            'difficulty': 'Medium',
            'expected_reward': '-200 to -100',
            'training_time': '~2-3 minutes'
        },
        'MountainCarContinuous-v0': {
            'description': 'Classic control task - drive a car up a hill',
            'action_space': 'Continuous (1 dimension)',
            'observation_space': 'Box (2 dimensions)',
            'difficulty': 'Medium',
            'expected_reward': '90+',
            'training_time': '~3-5 minutes'
        },
        'LunarLander-v2': {
            'description': 'Box2D task - land a spacecraft safely',
            'action_space': 'Discrete (4 actions)',
            'observation_space': 'Box (8 dimensions)',
            'difficulty': 'Hard',
            'expected_reward': '200+',
            'training_time': '~5-10 minutes'
        }
    }
    return env_info.get(env_name, {})


def train_ppo(args):
    """Train PPO agent."""
    print("="*70)
    print("Quick PPO Test - Simplified Environment")
    print("="*70)
    
    env_info = get_env_info(args.env)
    print(f"\nEnvironment: {args.env}")
    print(f"Description: {env_info.get('description', 'N/A')}")
    print(f"Action Space: {env_info.get('action_space', 'N/A')}")
    print(f"Observation Space: {env_info.get('observation_space', 'N/A')}")
    print(f"Difficulty: {env_info.get('difficulty', 'N/A')}")
    print(f"Expected Reward: {env_info.get('expected_reward', 'N/A')}")
    print(f"Estimated Training Time: {env_info.get('training_time', 'N/A')}")
    
    print(f"\nTraining Parameters:")
    print(f"  Total Timesteps: {args.total_timesteps}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Steps per Update: {args.n_steps}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Seed: {args.seed}")
    
    print("\n" + "-"*70)
    print("Step 1: Creating environment...")
    
    env = gym.make(args.env)
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    print("\nStep 2: Creating PPO agent...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log = os.path.join(args.output_dir, f"tb_{args.env}_{timestamp}")
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=args.seed,
    )
    
    print(f"\nStep 3: Training for {args.total_timesteps} timesteps...")
    print("  This may take a few minutes...")
    print("-"*70)
    
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=args.total_timesteps,
        log_interval=1,
        tb_log_name=f"PPO_{args.env}",
        progress_bar=True,
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\nTraining completed in {training_time:.2f} seconds!")
    
    print("\nStep 4: Evaluating trained model...")
    print(f"  Running {args.eval_episodes} evaluation episodes...")
    
    eval_env = gym.make(args.env, render_mode="human" if args.render else None)
    
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=args.render,
    )
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Expected Reward: {env_info.get('expected_reward', 'N/A')}")
    
    if mean_reward > 0:
        if args.env == 'CartPole-v1' and mean_reward >= 400:
            print("  ✅ Excellent! CartPole is solved (reward >= 400)")
        elif args.env == 'Pendulum-v1' and mean_reward >= -150:
            print("  ✅ Good performance on Pendulum!")
        elif args.env == 'MountainCarContinuous-v0' and mean_reward >= 80:
            print("  ✅ MountainCar solved!")
        elif args.env == 'LunarLander-v2' and mean_reward >= 200:
            print("  ✅ LunarLander solved!")
        else:
            print("  ⚠️  Training may need more timesteps or hyperparameter tuning")
    else:
        print("  ⚠️  Model still learning, try more timesteps")
    
    if args.save_model:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f"ppo_{args.env}_{timestamp}.zip")
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")
    
    env.close()
    eval_env.close()
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Environment: {args.env}")
    print(f"Algorithm: PPO")
    print(f"Total Timesteps: {args.total_timesteps}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"TensorBoard Logs: {tensorboard_log}")
    print("="*70)
    
    print("\nNext Steps:")
    print("1. View TensorBoard: tensorboard --logdir", tensorboard_log)
    print("2. Try different environment: --env Pendulum-v1")
    print("3. Increase training: --total_timesteps 100000")
    print("4. Render evaluation: --render")
    print("5. Save model: --save_model")
    
    return model, mean_reward, std_reward


if __name__ == "__main__":
    args = parse_args()
    train_ppo(args)
