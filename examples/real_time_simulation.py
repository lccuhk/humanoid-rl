#!/usr/bin/env python3
"""
Real-time simulation demo for trained humanoid robot.

This script runs a real-time simulation with rendering to visualize
the trained robot's walking and running policies.
"""

import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments import HumanoidMuJoCoEnv
from src.agents import PPOAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time simulation demo for trained humanoid robot'
    )
    
    parser.add_argument(
        '--walk_model', type=str,
        default='./checkpoints/ppo_mujoco_walk_20260522_120133_final.zip',
        help='Path to walking model checkpoint'
    )
    
    parser.add_argument(
        '--run_model', type=str,
        default='./checkpoints/ppo_mujoco_run_optimized_20260531_200649_1400000_steps.zip',
        help='Path to running model checkpoint'
    )
    
    parser.add_argument(
        '--task', type=str, default='run',
        choices=['walk', 'run', 'both'],
        help='Task to simulate (default: run)'
    )
    
    parser.add_argument(
        '--n_episodes', type=int, default=5,
        help='Number of episodes to simulate (default: 5)'
    )
    
    parser.add_argument(
        '--max_steps', type=int, default=1000,
        help='Maximum steps per episode (default: 1000)'
    )
    
    parser.add_argument(
        '--speed', type=float, default=1.0,
        help='Simulation speed multiplier (default: 1.0)'
    )
    
    parser.add_argument(
        '--show_stats', action='store_true',
        default=True,
        help='Show real-time statistics overlay'
    )
    
    parser.add_argument(
        '--velocity_bonus_weight', type=float, default=30.0,
        help='Velocity bonus weight for optimized run env'
    )
    
    parser.add_argument(
        '--air_time_reward_weight', type=float, default=50.0,
        help='Air time reward weight for optimized run env'
    )
    
    parser.add_argument(
        '--energy_efficiency_weight', type=float, default=5.0,
        help='Energy efficiency weight for optimized run env'
    )
    
    parser.add_argument(
        '--healthy_reward', type=float, default=1.0,
        help='Healthy reward weight (default: 1.0)'
    )
    
    parser.add_argument(
        '--fall_penalty', type=float, default=50.0,
        help='Fall penalty (default: 50.0)'
    )
    
    return parser.parse_args()


def simulate_episode(env, agent, max_steps, task_name, speed, show_stats, episode_num):
    """
    Simulate a single episode with real-time rendering.
    
    Returns:
        Dictionary with episode statistics
    """
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0
    max_velocity = 0.0
    start_x = info.get('x_position', 0.0)
    
    print(f"\n{'='*70}")
    print(f"Episode {episode_num} - {task_name.upper()}")
    print(f"{'='*70}")
    
    step_times = []
    
    while not done and step_count < max_steps:
        step_start = time.time()
        
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        total_reward += float(reward)
        step_count += 1
        
        current_vel = info.get('x_velocity', 0.0)
        max_velocity = max(max_velocity, abs(current_vel))
        current_x = info.get('x_position', start_x)
        distance = current_x - start_x
        
        if show_stats and step_count % 30 == 0:
            print(f"  Step {step_count:4d} | "
                  f"Reward: {total_reward:8.2f} | "
                  f"Velocity: {current_vel:6.2f} m/s | "
                  f"Distance: {distance:6.2f} m | "
                  f"Height: {info.get('z_position', 0):5.2f} m | "
                  f"Air Time: {info.get('air_time', 0):5.3f}s | "
                  f"Foot Contact: {'Yes' if info.get('foot_contact', True) else 'No'}")
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        target_step_time = 0.01 / speed
        if step_time < target_step_time:
            time.sleep(target_step_time - step_time)
    
    final_distance = info.get('x_position', start_x) - start_x
    avg_step_time = np.mean(step_times) if step_times else 0.0
    
    print(f"\nEpisode {episode_num} Complete:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Steps: {step_count}")
    print(f"  Distance Traveled: {final_distance:.2f} m")
    print(f"  Max Velocity: {max_velocity:.2f} m/s")
    print(f"  Average Step Time: {avg_step_time*1000:.2f} ms")
    print(f"  Reason: {'Terminated (fell)' if terminated else 'Truncated (max steps)' if truncated else 'Complete'}")
    
    return {
        'episode': episode_num,
        'task': task_name,
        'total_reward': total_reward,
        'steps': step_count,
        'distance': final_distance,
        'max_velocity': max_velocity,
        'avg_step_time_ms': avg_step_time * 1000,
        'terminated': terminated,
        'truncated': truncated
    }


def simulate_task(task, model_path, n_episodes, max_steps, speed, show_stats,
                  velocity_bonus_weight, air_time_reward_weight, energy_efficiency_weight,
                  healthy_reward, fall_penalty):
    """Simulate multiple episodes for a specific task."""
    print("\n" + "="*70)
    print(f"REAL-TIME SIMULATION: {task.upper()} TASK")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Simulation speed: {speed}x")
    print(f"Show statistics: {show_stats}")
    print(f"Healthy reward: {healthy_reward}")
    print(f"Fall penalty: {fall_penalty}")
    print("="*70)
    
    if task == 'run':
        env = HumanoidMuJoCoEnv(
            render_mode='human',
            task='run',
            forward_reward_weight=5.0,
            healthy_reward=healthy_reward,
            backward_penalty_weight=10.0,
            position_reward_weight=2.0,
            velocity_bonus_weight=velocity_bonus_weight,
            air_time_reward_weight=air_time_reward_weight,
            energy_efficiency_weight=energy_efficiency_weight,
            fall_penalty=fall_penalty,
        )
    else:
        env = HumanoidMuJoCoEnv(
            render_mode='human',
            task='walk',
            forward_reward_weight=5.0,
            healthy_reward=healthy_reward,
            backward_penalty_weight=10.0,
            position_reward_weight=2.0,
        )
    
    print("\nLoading model...")
    agent = PPOAgent(env=env)
    agent.load(model_path, env=env)
    print("Model loaded successfully!")
    
    print(f"\nStarting simulation in 3 seconds... (press Ctrl+C to stop)")
    time.sleep(3)
    
    results = []
    try:
        for episode in range(1, n_episodes + 1):
            result = simulate_episode(
                env, agent, max_steps, task, speed, show_stats, episode
            )
            results.append(result)
            
            if episode < n_episodes:
                print(f"\nNext episode in 2 seconds...")
                time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")
    
    finally:
        env.close()
    
    print("\n" + "="*70)
    print(f"{task.upper()} SIMULATION SUMMARY")
    print("="*70)
    if results:
        print(f"Mean Reward: {np.mean([r['total_reward'] for r in results]):.2f}")
        print(f"Mean Steps: {np.mean([r['steps'] for r in results]):.1f}")
        print(f"Mean Distance: {np.mean([r['distance'] for r in results]):.2f} m")
        print(f"Mean Max Velocity: {np.mean([r['max_velocity'] for r in results]):.2f} m/s")
        print(f"Success Rate: {sum(1 for r in results if r['truncated'] and not r['terminated']) / len(results) * 100:.1f}%")
    print("="*70)
    
    return results


def main():
    """Main function."""
    args = parse_args()
    
    print("="*70)
    print("HUMANOID ROBOT REAL-TIME SIMULATION")
    print("="*70)
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    all_results = []
    
    try:
        if args.task in ['walk', 'both']:
            walk_results = simulate_task(
                'walk', args.walk_model, args.n_episodes, args.max_steps,
                args.speed, args.show_stats,
                args.velocity_bonus_weight, args.air_time_reward_weight,
                args.energy_efficiency_weight,
                args.healthy_reward, args.fall_penalty
            )
            all_results.extend(walk_results)
            
            if args.task == 'both':
                print("\n" + "="*70)
                print("Switching to RUN task in 5 seconds...")
                print("="*70)
                time.sleep(5)
        
        if args.task in ['run', 'both']:
            run_results = simulate_task(
                'run', args.run_model, args.n_episodes, args.max_steps,
                args.speed, args.show_stats,
                args.velocity_bonus_weight, args.air_time_reward_weight,
                args.energy_efficiency_weight,
                args.healthy_reward, args.fall_penalty
            )
            all_results.extend(run_results)
    
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"Total episodes simulated: {len(all_results)}")
    print("="*70)


if __name__ == "__main__":
    main()
