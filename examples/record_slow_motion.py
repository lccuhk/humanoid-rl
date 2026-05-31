#!/usr/bin/env python3
"""
Record slow-motion videos of trained humanoid robot running.

This script records high-frame-rate videos for slow-motion playback,
allowing detailed observation of the flight phase during running.
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
        description='Record slow-motion videos of humanoid robot running'
    )
    
    parser.add_argument(
        '--model', type=str,
        default='./checkpoints/ppo_mujoco_run_20260531_170050_final.zip',
        help='Path to running model checkpoint'
    )
    
    parser.add_argument(
        '--n_episodes', type=int, default=2,
        help='Number of episodes to record (default: 2)'
    )
    
    parser.add_argument(
        '--max_steps', type=int, default=200,
        help='Maximum steps per episode (default: 200)'
    )
    
    parser.add_argument(
        '--slowdown_factor', type=float, default=0.5,
        help='Slowdown factor (default: 0.5 for half speed)'
    )
    
    parser.add_argument(
        '--output_dir', type=str,
        default='./logs/videos',
        help='Output directory for videos (default: ./logs/videos)'
    )
    
    parser.add_argument(
        '--fps', type=int, default=60,
        help='Video FPS (default: 60 for smooth slow motion)'
    )
    
    parser.add_argument(
        '--width', type=int, default=1920,
        help='Video width (default: 1920)'
    )
    
    parser.add_argument(
        '--height', type=int, default=1080,
        help='Video height (default: 1080)'
    )
    
    parser.add_argument(
        '--velocity_bonus_weight', type=float, default=30.0,
        help='Velocity bonus weight'
    )
    
    parser.add_argument(
        '--air_time_reward_weight', type=float, default=50.0,
        help='Air time reward weight'
    )
    
    parser.add_argument(
        '--energy_efficiency_weight', type=float, default=5.0,
        help='Energy efficiency weight'
    )
    
    return parser.parse_args()


def record_slow_motion_episode(
    env,
    agent,
    max_steps: int,
    output_dir: str,
    fps: int,
    width: int,
    height: int,
    slowdown_factor: float,
    episode_num: int
) -> Dict[str, Any]:
    """
    Record a single episode with detailed step information for slow motion.
    
    Returns:
        Dictionary with episode statistics
    """
    import cv2
    
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0
    max_velocity = 0.0
    start_x = info.get('x_position', 0.0)
    
    frames = []
    step_details = []
    
    print(f"\n{'='*80}")
    print(f"Slow-Motion Episode {episode_num}")
    print(f"{'='*80}")
    print(f"{'Step':>6} | {'Reward':>10} | {'Velocity':>10} | {'Distance':>10} | {'Height':>8} | {'AirTime':>10} | {'Contact':>8} | {'Phase':>12}")
    print(f"{'-'*80}")
    
    while not done and step_count < max_steps:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        total_reward += float(reward)
        step_count += 1
        
        current_vel = info.get('x_velocity', 0.0)
        max_velocity = max(max_velocity, abs(current_vel))
        current_x = info.get('x_position', start_x)
        distance = current_x - start_x
        z_height = info.get('z_position', 0.0)
        air_time = info.get('air_time', 0.0)
        foot_contact = info.get('foot_contact', True)
        is_healthy = info.get('is_healthy', True)
        
        if air_time > 0.01 and not foot_contact:
            phase = "FLIGHT 🏃‍♂️"
        elif foot_contact and air_time > 0.01:
            phase = "LANDING 👟"
        else:
            phase = "STANCE 🦶"
        
        if step_count % 5 == 0 or air_time > 0.05:
            print(f"{step_count:>6} | {total_reward:>10.2f} | {current_vel:>9.2f}m/s | {distance:>9.2f}m | {z_height:>7.2f}m | {air_time:>9.3f}s | {'Yes' if foot_contact else 'No':>8} | {phase:>12}")
        
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        step_details.append({
            'step': step_count,
            'reward': float(reward),
            'total_reward': total_reward,
            'velocity': current_vel,
            'distance': distance,
            'z_height': z_height,
            'air_time': air_time,
            'foot_contact': foot_contact,
            'is_healthy': is_healthy,
            'phase': phase
        })
    
    final_distance = info.get('x_position', start_x) - start_x
    
    print(f"\n{'-'*80}")
    print(f"Episode {episode_num} Complete:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Steps: {step_count}")
    print(f"  Distance Traveled: {final_distance:.2f} m")
    print(f"  Max Velocity: {max_velocity:.2f} m/s")
    print(f"  Reason: {'Terminated (fell)' if terminated else 'Truncated (max steps)' if truncated else 'Complete'}")
    
    flight_phases = [s for s in step_details if 'FLIGHT' in s['phase']]
    if flight_phases:
        max_flight_time = max(s['air_time'] for s in flight_phases)
        avg_flight_time = np.mean([s['air_time'] for s in flight_phases])
        flight_count = sum(1 for i, s in enumerate(step_details) if i > 0 and 'FLIGHT' in s['phase'] and 'FLIGHT' not in step_details[i-1]['phase'])
        print(f"\nFlight Phase Analysis:")
        print(f"  Number of flight phases: {flight_count}")
        print(f"  Max flight time: {max_flight_time:.3f} s")
        print(f"  Average flight time: {avg_flight_time:.3f} s")
    
    if frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"run_slowmotion_{slowdown_factor}x_episode{episode_num}_{timestamp}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        
        output_fps = int(fps * slowdown_factor)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, output_fps, (width, height))
        
        for frame in frames:
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            
            video_writer.write(frame)
        
        video_writer.release()
        
        print(f"\nVideo saved to: {video_path}")
        print(f"Playback speed: {slowdown_factor}x (original {fps}fps -> playback {output_fps}fps)")
        print(f"Duration: {len(frames) / fps:.2f}s (original) / {len(frames) / output_fps:.2f}s (slow motion)")
    
    return {
        'episode': episode_num,
        'total_reward': total_reward,
        'steps': step_count,
        'distance': final_distance,
        'max_velocity': max_velocity,
        'flight_phases': len([s for s in step_details if 'FLIGHT' in s['phase']]),
        'max_flight_time': max([s['air_time'] for s in step_details]) if step_details else 0,
        'terminated': terminated,
        'truncated': truncated,
        'step_details': step_details
    }


def main():
    """Main function."""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("HUMANOID ROBOT SLOW-MOTION RUNNING RECORDING")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Slowdown factor: {args.slowdown_factor}x")
    print(f"Recording FPS: {args.fps}")
    print(f"Playback FPS: {int(args.fps * args.slowdown_factor)}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    print("\nCreating environment...")
    env = HumanoidMuJoCoEnv(
        render_mode='rgb_array',
        task='run',
        forward_reward_weight=5.0,
        healthy_reward=1.0,
        backward_penalty_weight=10.0,
        position_reward_weight=2.0,
        velocity_bonus_weight=args.velocity_bonus_weight,
        air_time_reward_weight=args.air_time_reward_weight,
        energy_efficiency_weight=args.energy_efficiency_weight,
    )
    
    print("Loading model...")
    agent = PPOAgent(env=env)
    agent.load(args.model, env=env)
    print("Model loaded successfully!")
    
    print(f"\nStarting recording in 3 seconds...")
    import time
    time.sleep(3)
    
    all_results = []
    try:
        for episode in range(1, args.n_episodes + 1):
            result = record_slow_motion_episode(
                env, agent, args.max_steps, args.output_dir,
                args.fps, args.width, args.height, args.slowdown_factor, episode
            )
            all_results.append(result)
            
            if episode < args.n_episodes:
                print(f"\nNext episode in 2 seconds...")
                time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\nRecording stopped by user.")
    
    finally:
        env.close()
    
    print("\n" + "="*80)
    print("SLOW-MOTION RECORDING SUMMARY")
    print("="*80)
    if all_results:
        print(f"Mean Reward: {np.mean([r['total_reward'] for r in all_results]):.2f}")
        print(f"Mean Steps: {np.mean([r['steps'] for r in all_results]):.1f}")
        print(f"Mean Distance: {np.mean([r['distance'] for r in all_results]):.2f} m")
        print(f"Mean Max Velocity: {np.mean([r['max_velocity'] for r in all_results]):.2f} m/s")
        print(f"Mean Flight Phases: {np.mean([r['flight_phases'] for r in all_results]):.1f}")
        print(f"Mean Max Flight Time: {np.mean([r['max_flight_time'] for r in all_results]):.3f} s")
    print("="*80)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'slowdown_factor': args.slowdown_factor,
        'fps': args.fps,
        'resolution': f"{args.width}x{args.height}",
        'episodes': all_results
    }
    
    summary_path = os.path.join(args.output_dir, f"slow_motion_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("RECORDING COMPLETE!")
    print("="*80)
    print(f"Videos saved to: {args.output_dir}")
    print(f"Play videos with: cd {args.output_dir} && python3 -m http.server 8888")
    print("="*80)


if __name__ == "__main__":
    main()
