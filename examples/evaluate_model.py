#!/usr/bin/env python3
"""
Example: Evaluate Trained Model

This script demonstrates how to evaluate a trained humanoid robot model.
"""

import os
import sys
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.environments import HumanoidMuJoCoEnv, HumanoidPyBulletEnv
from src.agents import PPOAgent, SACAgent, TD3Agent
from src.visualization import RobotVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained humanoid robot model'
    )
    
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--env', type=str, default='mujoco',
        choices=['mujoco', 'pybullet'],
        help='Physics engine (default: mujoco)'
    )
    
    parser.add_argument(
        '--algo', type=str, default='ppo',
        choices=['ppo', 'sac', 'td3'],
        help='RL algorithm (default: ppo)'
    )
    
    parser.add_argument(
        '--task', type=str, default='walk',
        choices=['walk', 'run'],
        help='Task type (default: walk)'
    )
    
    parser.add_argument(
        '--n_episodes', type=int, default=10,
        help='Number of evaluation episodes (default: 10)'
    )
    
    parser.add_argument(
        '--render', action='store_true',
        help='Render environment during evaluation'
    )
    
    parser.add_argument(
        '--record', action='store_true',
        help='Record video of evaluation'
    )
    
    parser.add_argument(
        '--output_dir', type=str, default='./logs/evaluation',
        help='Output directory for results (default: ./logs/evaluation)'
    )
    
    return parser.parse_args()


def create_environment(args):
    """Create evaluation environment."""
    render_mode = 'human' if args.render else 'rgb_array' if args.record else None
    
    if args.env == 'mujoco':
        env = HumanoidMuJoCoEnv(
            render_mode=render_mode,
            task=args.task,
        )
    elif args.env == 'pybullet':
        env = HumanoidPyBulletEnv(
            render_mode=render_mode,
            task=args.task,
        )
    else:
        raise ValueError(f"Unknown environment: {args.env}")
    
    return env


def load_agent(args, env):
    """Load trained agent."""
    if args.algo == 'ppo':
        agent = PPOAgent(env=env)
    elif args.algo == 'sac':
        agent = SACAgent(env=env)
    elif args.algo == 'td3':
        agent = TD3Agent(env=env)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")
    
    agent.load(args.model_path, env=env)
    return agent


def main():
    args = parse_args()
    
    print("="*70)
    print("Humanoid Robot Model Evaluation")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")
    print(f"Task: {args.task}")
    print(f"Episodes: {args.n_episodes}")
    print("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n1. Creating environment...")
    env = create_environment(args)
    
    print("2. Loading trained model...")
    agent = load_agent(args, env)
    
    print("3. Setting up visualizer...")
    visualizer = RobotVisualizer(
        output_dir=args.output_dir,
    )
    
    print(f"\n4. Running evaluation for {args.n_episodes} episodes...")
    
    all_rewards = []
    all_lengths = []
    all_forward_distances = []
    all_max_velocities = []
    
    for episode in range(args.n_episodes):
        print(f"\nEpisode {episode + 1}/{args.n_episodes}")
        
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        max_velocity = 0
        start_pos = info.get('x_position', 0)
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if 'x_velocity' in info:
                max_velocity = max(max_velocity, info['x_velocity'])
            
            if hasattr(env, 'get_state'):
                state = env.get_state()
                visualizer.log_state(state)
                visualizer.log_action(action)
            
            done = terminated or truncated
        
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        all_max_velocities.append(max_velocity)
        
        if 'x_position' in info:
            forward_distance = info['x_position'] - start_pos
            all_forward_distances.append(forward_distance)
            print(f"   Reward: {episode_reward:.2f}, Length: {episode_length}, "
                  f"Distance: {forward_distance:.2f}m, Max Vel: {max_velocity:.2f}m/s")
        else:
            print(f"   Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    
    results = {
        'mean_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'min_reward': float(np.min(all_rewards)),
        'max_reward': float(np.max(all_rewards)),
        'mean_episode_length': float(np.mean(all_lengths)),
        'std_episode_length': float(np.std(all_lengths)),
    }
    
    if all_forward_distances:
        results['mean_forward_distance'] = float(np.mean(all_forward_distances))
        results['std_forward_distance'] = float(np.std(all_forward_distances))
    
    if all_max_velocities:
        results['mean_max_velocity'] = float(np.mean(all_max_velocities))
        results['std_max_velocity'] = float(np.std(all_max_velocities))
    
    print("\nSummary Statistics:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n5. Generating visualizations...")
    visualizer.create_robot_dashboard(save=True)
    visualizer.plot_trajectory(save=True)
    visualizer.plot_velocity_profile(save=True)
    visualizer.plot_gait_analysis(save=True)
    
    print("\n6. Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.json")
    
    full_results = {
        'model_path': args.model_path,
        'environment': args.env,
        'algorithm': args.algo,
        'task': args.task,
        'n_episodes': args.n_episodes,
        'timestamp': timestamp,
        'summary': results,
        'episode_rewards': all_rewards,
        'episode_lengths': all_lengths,
    }
    
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"   Results saved to: {results_path}")
    
    env.close()
    
    print("\n" + "="*70)
    print("Evaluation completed!")
    print("="*70)


if __name__ == "__main__":
    main()
