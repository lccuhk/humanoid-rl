#!/usr/bin/env python3
"""
Record demo videos of trained humanoid robot.

This script records videos of the trained robot performing
walking and running tasks.
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments import HumanoidMuJoCoEnv
from src.agents import PPOAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Record demo videos of trained humanoid robot'
    )
    
    parser.add_argument(
        '--walk_model', type=str,
        default='./checkpoints/ppo_mujoco_walk_20260522_120133_final.zip',
        help='Path to walking model checkpoint'
    )
    
    parser.add_argument(
        '--run_model', type=str,
        default='./checkpoints/ppo_mujoco_run_20260531_170050_final.zip',
        help='Path to running model checkpoint'
    )
    
    parser.add_argument(
        '--task', type=str, default='both',
        choices=['walk', 'run', 'both'],
        help='Task to record (default: both)'
    )
    
    parser.add_argument(
        '--n_episodes', type=int, default=3,
        help='Number of episodes to record per task (default: 3)'
    )
    
    parser.add_argument(
        '--max_steps', type=int, default=500,
        help='Maximum steps per episode (default: 500)'
    )
    
    parser.add_argument(
        '--output_dir', type=str,
        default='./logs/videos',
        help='Output directory for videos (default: ./logs/videos)'
    )
    
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Video FPS (default: 30)'
    )
    
    parser.add_argument(
        '--width', type=int, default=1280,
        help='Video width (default: 1280)'
    )
    
    parser.add_argument(
        '--height', type=int, default=720,
        help='Video height (default: 720)'
    )
    
    return parser.parse_args()


def record_episode(
    env,
    agent,
    max_steps: int,
    task_name: str,
    output_dir: str,
    fps: int,
    width: int,
    height: int,
    episode_num: int
) -> Dict[str, Any]:
    """
    Record a single episode and save as video.
    
    Returns:
        Dictionary with episode statistics
    """
    import cv2
    
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    frames = []
    
    start_x = info.get('x_position', 0)
    
    while not done and step_count < max_steps:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = env.render()
        if frame is not None:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            frames.append(frame)
        
        total_reward += reward
        step_count += 1
        done = terminated or truncated
    
    final_x = info.get('x_position', start_x)
    distance = final_x - start_x
    max_vel = info.get('x_velocity', 0)
    
    if frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(
            output_dir,
            f"{task_name}_episode{episode_num}_{timestamp}.mp4"
        )
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for frame in frames:
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"  Saved video: {video_path}")
        
        return {
            'video_path': video_path,
            'total_reward': total_reward,
            'steps': step_count,
            'distance': distance,
            'max_velocity': max_vel,
            'frames': len(frames)
        }
    
    return None


def record_task(
    task: str,
    model_path: str,
    n_episodes: int,
    max_steps: int,
    output_dir: str,
    fps: int,
    width: int,
    height: int
) -> List[Dict[str, Any]]:
    """
    Record multiple episodes for a specific task.
    
    Returns:
        List of episode statistics
    """
    print(f"\n{'='*70}")
    print(f"Recording {task.upper()} task")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return []
    
    env = HumanoidMuJoCoEnv(
        render_mode='rgb_array',
        task=task,
    )
    
    agent = PPOAgent(env=env)
    agent.load(model_path, env=env)
    
    results = []
    
    for episode in range(n_episodes):
        print(f"Recording episode {episode + 1}/{n_episodes}...")
        
        result = record_episode(
            env, agent, max_steps, task, output_dir,
            fps, width, height, episode + 1
        )
        
        if result:
            results.append(result)
            print(f"  Reward: {result['total_reward']:.2f} | "
                  f"Steps: {result['steps']} | "
                  f"Distance: {result['distance']:.2f}m | "
                  f"Max Vel: {result['max_velocity']:.2f}m/s\n")
        else:
            print(f"  ⚠️  No frames recorded\n")
    
    env.close()
    
    if results:
        print(f"\n{'='*70}")
        print(f"{task.upper()} Summary")
        print(f"{'='*70}")
        print(f"Mean Reward: {np.mean([r['total_reward'] for r in results]):.2f}")
        print(f"Mean Steps: {np.mean([r['steps'] for r in results]):.1f}")
        print(f"Mean Distance: {np.mean([r['distance'] for r in results]):.2f}m")
        print(f"Mean Max Velocity: {np.mean([r['max_velocity'] for r in results]):.2f}m/s")
        print(f"{'='*70}")
    
    return results


def main():
    """Main function."""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Humanoid Robot Demo Video Recorder")
    print("="*70)
    print(f"Output Directory: {args.output_dir}")
    print(f"Video Resolution: {args.width}x{args.height}")
    print(f"Video FPS: {args.fps}")
    print("="*70)
    
    all_results = {}
    
    if args.task in ['walk', 'both']:
        walk_results = record_task(
            'walk', args.walk_model, args.n_episodes, args.max_steps,
            args.output_dir, args.fps, args.width, args.height
        )
        all_results['walk'] = walk_results
    
    if args.task in ['run', 'both']:
        run_results = record_task(
            'run', args.run_model, args.n_episodes, args.max_steps,
            args.output_dir, args.fps, args.width, args.height
        )
        all_results['run'] = run_results
    
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(args.output_dir, f"recording_summary_{timestamp}.json")
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Recording complete!")
        print(f"Summary saved to: {summary_path}")
        print(f"Videos saved to: {args.output_dir}")
        print("="*70)


if __name__ == "__main__":
    main()
