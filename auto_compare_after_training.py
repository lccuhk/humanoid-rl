#!/usr/bin/env python3
"""
Auto-compare script that waits for training to complete and then
generates comparison plots between the optimized model and baseline model.
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Wait for training to complete and auto-generate comparison plots'
    )
    
    parser.add_argument(
        '--training_log', type=str,
        default='./logs/training/ppo_mujoco_run_optimized_20260531_203046',
        help='Path to training log directory'
    )
    
    parser.add_argument(
        '--baseline_log', type=str,
        default=None,
        help='Path to baseline training log directory (optional)'
    )
    
    parser.add_argument(
        '--skip_training_comparison', action='store_true',
        default=False,
        help='Skip training curve comparison if baseline log is not available'
    )
    
    parser.add_argument(
        '--checkpoint_dir', type=str,
        default='./checkpoints',
        help='Directory containing model checkpoints'
    )
    
    parser.add_argument(
        '--output_dir', type=str,
        default='./logs/comparison',
        help='Output directory for comparison results'
    )
    
    parser.add_argument(
        '--total_timesteps', type=int,
        default=2000000,
        help='Total training timesteps to wait for'
    )
    
    parser.add_argument(
        '--check_interval', type=int,
        default=60,
        help='Check interval in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--timeout', type=int,
        default=7200,
        help='Timeout in seconds (default: 7200 = 2 hours)'
    )
    
    parser.add_argument(
        '--baseline_model', type=str,
        default='./checkpoints/ppo_mujoco_run_20260531_170050_final.zip',
        help='Path to baseline model checkpoint'
    )
    
    parser.add_argument(
        '--optimized_model_prefix', type=str,
        default='ppo_mujoco_run_optimized_20260531_203046',
        help='Prefix for optimized model checkpoints'
    )
    
    parser.add_argument(
        '--n_eval_episodes', type=int,
        default=5,
        help='Number of evaluation episodes per model'
    )
    
    return parser.parse_args()


def wait_for_training_completion(
    training_log_dir: str,
    total_timesteps: int,
    check_interval: int,
    timeout: int
) -> bool:
    """
    Wait for training to complete by monitoring the training log.
    
    Returns:
        True if training completed successfully, False if timeout
    """
    print("="*80)
    print("WAITING FOR TRAINING COMPLETION")
    print("="*80)
    print(f"Training log directory: {training_log_dir}")
    print(f"Total timesteps expected: {total_timesteps:,}")
    print(f"Check interval: {check_interval}s")
    print(f"Timeout: {timeout}s ({timeout/3600:.1f} hours)")
    print("="*80)
    
    start_time = time.time()
    last_timesteps = 0
    
    while time.time() - start_time < timeout:
        try:
            log_files = sorted([
                f for f in os.listdir(training_log_dir)
                if f.endswith('.csv')
            ])
            
            if log_files:
                latest_log = os.path.join(training_log_dir, log_files[-1])
                df = pd.read_csv(latest_log)
                current_timesteps = len(df)
                
                if current_timesteps > last_timesteps:
                    elapsed = time.time() - start_time
                    progress = current_timesteps / total_timesteps * 100
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Progress: {current_timesteps:,}/{total_timesteps:,} "
                          f"({progress:.1f}%) - Elapsed: {elapsed/60:.1f}min")
                    last_timesteps = current_timesteps
                
                if current_timesteps >= total_timesteps * 0.99:
                    print("\n✅ Training appears to be complete!")
                    return True
            
            time.sleep(check_interval)
            
        except Exception as e:
            print(f"Error checking training progress: {e}")
            time.sleep(check_interval)
    
    print("\n❌ Timeout waiting for training completion")
    return False


def find_final_model(checkpoint_dir: str, model_prefix: str) -> str:
    """Find the final model checkpoint."""
    import glob
    
    # Look for final model first
    final_pattern = os.path.join(checkpoint_dir, f"{model_prefix}_final.zip")
    final_files = glob.glob(final_pattern)
    if final_files:
        return final_files[0]
    
    # Look for interrupted model
    interrupted_pattern = os.path.join(checkpoint_dir, f"{model_prefix}_interrupted.zip")
    interrupted_files = glob.glob(interrupted_pattern)
    if interrupted_files:
        return interrupted_files[0]
    
    # Look for latest checkpoint
    checkpoint_pattern = os.path.join(checkpoint_dir, f"{model_prefix}_*_steps.zip")
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))
    if checkpoint_files:
        return checkpoint_files[-1]
    
    return None


def evaluate_model(
    model_path: str,
    n_episodes: int,
    max_steps: int = 500
) -> List[Dict[str, Any]]:
    """
    Evaluate a model and return episode statistics.
    """
    from src.environments import HumanoidMuJoCoEnv
    from src.agents import PPOAgent
    
    print(f"\nEvaluating model: {os.path.basename(model_path)}")
    
    env = HumanoidMuJoCoEnv(
        render_mode=None,
        task='run',
        forward_reward_weight=5.0,
        healthy_reward=1.0,
        backward_penalty_weight=10.0,
        position_reward_weight=2.0,
        velocity_bonus_weight=30.0,
        air_time_reward_weight=50.0,
        energy_efficiency_weight=5.0,
    )
    
    agent = PPOAgent(env=env)
    agent.load(model_path, env=env)
    
    results = []
    
    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0.0
        max_velocity = 0.0
        start_x = info.get('x_position', 0.0)
        rewards = []
        
        while not done and step_count < max_steps:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            total_reward += float(reward)
            step_count += 1
            rewards.append(float(reward))
            
            current_vel = info.get('x_velocity', 0.0)
            max_velocity = max(max_velocity, abs(current_vel))
        
        final_distance = info.get('x_position', start_x) - start_x
        
        results.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': step_count,
            'distance': final_distance,
            'max_velocity': max_velocity,
            'mean_reward_per_step': np.mean(rewards) if rewards else 0,
            'terminated': terminated,
            'truncated': truncated
        })
        
        print(f"  Episode {episode}: Reward={total_reward:.2f}, Steps={step_count}, "
              f"Distance={final_distance:.2f}m, MaxVel={max_velocity:.2f}m/s")
    
    env.close()
    return results


def load_training_data(log_dir: str) -> pd.DataFrame:
    """Load training data from log directory."""
    import glob
    
    log_files = sorted([
        f for f in os.listdir(log_dir)
        if f.endswith('.csv')
    ])
    
    if not log_files:
        return None
    
    dfs = []
    for log_file in log_files:
        df = pd.read_csv(os.path.join(log_dir, log_file))
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def compute_episode_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute episode-level statistics from step-level data."""
    episode_stats = df.groupby('episode').agg({
        'reward': ['sum', 'mean', 'min', 'max', 'count'],
        'reward_forward': 'sum',
        'reward_position': 'sum',
        'reward_velocity_bonus': 'sum',
        'reward_air_time': 'sum',
        'reward_energy_efficiency': 'sum',
        'x_velocity': 'mean',
        'distance_traveled': 'last',
    }).reset_index()
    
    episode_stats.columns = [
        'episode', 'reward_sum', 'reward_mean', 'reward_min', 'reward_max', 'steps',
        'reward_forward_sum', 'reward_position_sum', 'reward_velocity_bonus_sum',
        'reward_air_time_sum', 'reward_energy_efficiency_sum',
        'x_velocity_mean', 'distance_traveled'
    ]
    
    return episode_stats


def plot_training_comparison(
    optimized_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    output_dir: str,
    window_size: int = 50
):
    """Generate comparison plots between optimized and baseline training."""
    os.makedirs(output_dir, exist_ok=True)
    
    opt_stats = compute_episode_statistics(optimized_df)
    base_stats = compute_episode_statistics(baseline_df)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('Training Comparison: Optimized vs Baseline', fontsize=16, fontweight='bold')
    
    # 1. Episode reward (raw)
    ax = axes[0, 0]
    ax.plot(opt_stats['episode'], opt_stats['reward_sum'], 
            label='Optimized', alpha=0.6, color='#2E86AB')
    ax.plot(base_stats['episode'], base_stats['reward_sum'], 
            label='Baseline', alpha=0.6, color='#A23B72')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards (Raw)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Episode reward (smoothed)
    ax = axes[0, 1]
    opt_smooth = opt_stats['reward_sum'].rolling(window=window_size, min_periods=1).mean()
    base_smooth = base_stats['reward_sum'].rolling(window=window_size, min_periods=1).mean()
    opt_std = opt_stats['reward_sum'].rolling(window=window_size, min_periods=1).std()
    base_std = base_stats['reward_sum'].rolling(window=window_size, min_periods=1).std()
    
    ax.plot(opt_stats['episode'], opt_smooth, label='Optimized', color='#2E86AB', linewidth=2)
    ax.fill_between(opt_stats['episode'], opt_smooth - opt_std, opt_smooth + opt_std, 
                    alpha=0.2, color='#2E86AB')
    ax.plot(base_stats['episode'], base_smooth, label='Baseline', color='#A23B72', linewidth=2)
    ax.fill_between(base_stats['episode'], base_smooth - base_std, base_smooth + base_std, 
                    alpha=0.2, color='#A23B72')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(f'Episode Rewards (Smoothed, window={window_size})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Episode length
    ax = axes[1, 0]
    opt_len_smooth = opt_stats['steps'].rolling(window=window_size, min_periods=1).mean()
    base_len_smooth = base_stats['steps'].rolling(window=window_size, min_periods=1).mean()
    ax.plot(opt_stats['episode'], opt_len_smooth, label='Optimized', color='#2E86AB', linewidth=2)
    ax.plot(base_stats['episode'], base_len_smooth, label='Baseline', color='#A23B72', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps per Episode')
    ax.set_title('Episode Length (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Distance traveled
    ax = axes[1, 1]
    opt_dist_smooth = opt_stats['distance_traveled'].rolling(window=window_size, min_periods=1).mean()
    base_dist_smooth = base_stats['distance_traveled'].rolling(window=window_size, min_periods=1).mean()
    ax.plot(opt_stats['episode'], opt_dist_smooth, label='Optimized', color='#2E86AB', linewidth=2)
    ax.plot(base_stats['episode'], base_dist_smooth, label='Baseline', color='#A23B72', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance Traveled per Episode (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. X velocity
    ax = axes[2, 0]
    opt_vel_smooth = opt_stats['x_velocity_mean'].rolling(window=window_size, min_periods=1).mean()
    base_vel_smooth = base_stats['x_velocity_mean'].rolling(window=window_size, min_periods=1).mean()
    ax.plot(opt_stats['episode'], opt_vel_smooth, label='Optimized', color='#2E86AB', linewidth=2)
    ax.plot(base_stats['episode'], base_vel_smooth, label='Baseline', color='#A23B72', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean X Velocity (m/s)')
    ax.set_title('Mean X Velocity per Episode (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Reward components (optimized)
    ax = axes[2, 1]
    components = ['reward_forward_sum', 'reward_position_sum', 'reward_velocity_bonus_sum',
                  'reward_air_time_sum', 'reward_energy_efficiency_sum']
    labels = ['Forward', 'Position', 'Velocity', 'Air Time', 'Energy']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    for comp, label, color in zip(components, labels, colors):
        smooth = opt_stats[comp].rolling(window=window_size, min_periods=1).mean()
        ax.plot(opt_stats['episode'], smooth, label=label, color=color, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward Component')
    ax.set_title('Optimized Model - Reward Components (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Reward components (baseline)
    ax = axes[3, 0]
    for comp, label, color in zip(components, labels, colors):
        if comp in base_stats.columns:
            smooth = base_stats[comp].rolling(window=window_size, min_periods=1).mean()
            ax.plot(base_stats['episode'], smooth, label=label, color=color, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward Component')
    ax.set_title('Baseline Model - Reward Components (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Evaluation results comparison
    ax = axes[3, 1]
    # This will be filled after evaluation
    ax.text(0.5, 0.5, 'Evaluation results\nwill be shown here', 
            ha='center', va='center', fontsize=14, alpha=0.5)
    ax.set_title('Post-Training Evaluation Comparison')
    ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f'training_comparison_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining comparison plot saved to: {plot_path}")
    
    return plot_path


def plot_evaluation_comparison(
    optimized_results: List[Dict[str, Any]],
    baseline_results: List[Dict[str, Any]],
    output_dir: str
):
    """Generate comparison plots from evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Post-Training Evaluation: Optimized vs Baseline', fontsize=16, fontweight='bold')
    
    metrics = ['total_reward', 'distance', 'max_velocity', 'steps']
    titles = ['Total Reward per Episode', 'Distance Traveled per Episode', 
              'Max Velocity per Episode', 'Steps per Episode']
    ylabels = ['Total Reward', 'Distance (m)', 'Max Velocity (m/s)', 'Steps']
    
    for ax, metric, title, ylabel in zip(axes.flat, metrics, titles, ylabels):
        opt_values = [r[metric] for r in optimized_results]
        base_values = [r[metric] for r in baseline_results]
        
        x = np.arange(len(opt_values))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, opt_values, width, label='Optimized', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, base_values, width, label='Baseline', color='#A23B72', alpha=0.8)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Ep {i+1}' for i in range(len(x))])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f'evaluation_comparison_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation comparison plot saved to: {plot_path}")
    
    # Generate summary statistics
    summary = {
        'optimized': {
            'mean_reward': np.mean([r['total_reward'] for r in optimized_results]),
            'std_reward': np.std([r['total_reward'] for r in optimized_results]),
            'mean_distance': np.mean([r['distance'] for r in optimized_results]),
            'mean_velocity': np.mean([r['max_velocity'] for r in optimized_results]),
            'mean_steps': np.mean([r['steps'] for r in optimized_results]),
        },
        'baseline': {
            'mean_reward': np.mean([r['total_reward'] for r in baseline_results]),
            'std_reward': np.std([r['total_reward'] for r in baseline_results]),
            'mean_distance': np.mean([r['distance'] for r in baseline_results]),
            'mean_velocity': np.mean([r['max_velocity'] for r in baseline_results]),
            'mean_steps': np.mean([r['steps'] for r in baseline_results]),
        }
    }
    
    # Calculate improvements
    improvements = {}
    for key in summary['optimized']:
        opt_val = summary['optimized'][key]
        base_val = summary['baseline'][key]
        if base_val != 0:
            improvements[key] = (opt_val - base_val) / abs(base_val) * 100
        else:
            improvements[key] = 0
    
    summary['improvements'] = improvements
    
    # Save summary
    summary_path = os.path.join(output_dir, f'evaluation_summary_{timestamp}.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation summary saved to: {summary_path}")
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Metric':<25} {'Optimized':>15} {'Baseline':>15} {'Improvement':>15}")
    print("-"*80)
    for key in summary['optimized']:
        opt_val = summary['optimized'][key]
        base_val = summary['baseline'][key]
        imp = improvements[key]
        imp_str = f"{imp:+.1f}%" if imp != 0 else "N/A"
        print(f"{key:<25} {opt_val:>15.2f} {base_val:>15.2f} {imp_str:>15}")
    print("="*80)
    
    return plot_path, summary


def main():
    """Main function."""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("AUTO-COMPARE SCRIPT STARTED")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Step 1: Wait for training to complete
    training_complete = wait_for_training_completion(
        args.training_log, args.total_timesteps, args.check_interval, args.timeout
    )
    
    if not training_complete:
        print("\n⚠️  Training did not complete within timeout. Proceeding with latest checkpoint.")
    
    # Step 2: Find the final optimized model
    print("\n" + "="*80)
    print("FINDING FINAL MODEL")
    print("="*80)
    
    final_model = find_final_model(args.checkpoint_dir, args.optimized_model_prefix)
    
    if final_model is None:
        print("❌ Could not find final model!")
        return 1
    
    print(f"✅ Found final model: {os.path.basename(final_model)}")
    
    # Step 3: Load training data
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)
    
    optimized_df = load_training_data(args.training_log)
    baseline_df = None
    
    if optimized_df is None:
        print("❌ Could not load optimized training data!")
        return 1
    
    print(f"✅ Loaded {len(optimized_df):,} steps from optimized training")
    
    if args.baseline_log and not args.skip_training_comparison:
        baseline_df = load_training_data(args.baseline_log)
        if baseline_df is not None:
            print(f"✅ Loaded {len(baseline_df):,} steps from baseline training")
        else:
            print("⚠️  Could not load baseline training data. Skipping training comparison.")
            args.skip_training_comparison = True
    elif args.skip_training_comparison:
        print("ℹ️  Skipping training curve comparison as requested")
    else:
        print("ℹ️  No baseline log provided. Skipping training curve comparison.")
        args.skip_training_comparison = True
    
    # Step 4: Generate training comparison plots
    if not args.skip_training_comparison and baseline_df is not None:
        print("\n" + "="*80)
        print("GENERATING TRAINING COMPARISON PLOTS")
        print("="*80)
        training_plot = plot_training_comparison(optimized_df, baseline_df, args.output_dir)
    else:
        training_plot = None
        
        # Generate single training plot for optimized model
        print("\n" + "="*80)
        print("GENERATING OPTIMIZED MODEL TRAINING PLOTS")
        print("="*80)
        
        opt_stats = compute_episode_statistics(optimized_df)
        window_size = 50
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle('Optimized Model Training Progress', fontsize=16, fontweight='bold')
        
        # 1. Episode reward (raw)
        ax = axes[0, 0]
        ax.plot(opt_stats['episode'], opt_stats['reward_sum'], 
                label='Optimized', alpha=0.6, color='#2E86AB')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards (Raw)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Episode reward (smoothed)
        ax = axes[0, 1]
        opt_smooth = opt_stats['reward_sum'].rolling(window=window_size, min_periods=1).mean()
        opt_std = opt_stats['reward_sum'].rolling(window=window_size, min_periods=1).std()
        
        ax.plot(opt_stats['episode'], opt_smooth, label='Optimized', color='#2E86AB', linewidth=2)
        ax.fill_between(opt_stats['episode'], opt_smooth - opt_std, opt_smooth + opt_std, 
                        alpha=0.2, color='#2E86AB')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title(f'Episode Rewards (Smoothed, window={window_size})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Episode length
        ax = axes[1, 0]
        opt_len_smooth = opt_stats['steps'].rolling(window=window_size, min_periods=1).mean()
        ax.plot(opt_stats['episode'], opt_len_smooth, label='Optimized', color='#2E86AB', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps per Episode')
        ax.set_title('Episode Length (Smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Distance traveled
        ax = axes[1, 1]
        opt_dist_smooth = opt_stats['distance_traveled'].rolling(window=window_size, min_periods=1).mean()
        ax.plot(opt_stats['episode'], opt_dist_smooth, label='Optimized', color='#2E86AB', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Distance (m)')
        ax.set_title('Distance Traveled per Episode (Smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. X velocity
        ax = axes[2, 0]
        opt_vel_smooth = opt_stats['x_velocity_mean'].rolling(window=window_size, min_periods=1).mean()
        ax.plot(opt_stats['episode'], opt_vel_smooth, label='Optimized', color='#2E86AB', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean X Velocity (m/s)')
        ax.set_title('Mean X Velocity per Episode (Smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Reward components
        ax = axes[2, 1]
        components = ['reward_forward_sum', 'reward_position_sum', 'reward_velocity_bonus_sum',
                      'reward_air_time_sum', 'reward_energy_efficiency_sum']
        labels = ['Forward', 'Position', 'Velocity', 'Air Time', 'Energy']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
        
        for comp, label, color in zip(components, labels, colors):
            smooth = opt_stats[comp].rolling(window=window_size, min_periods=1).mean()
            ax.plot(opt_stats['episode'], smooth, label=label, color=color, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward Component')
        ax.set_title('Reward Components (Smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_plot = os.path.join(args.output_dir, f'optimized_training_progress_{timestamp}.png')
        plt.savefig(training_plot, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Optimized training progress plot saved to: {training_plot}")
    
    # Step 5: Evaluate both models
    print("\n" + "="*80)
    print("EVALUATING MODELS")
    print("="*80)
    
    print("\nEvaluating optimized model...")
    optimized_results = evaluate_model(final_model, args.n_eval_episodes)
    
    print("\nEvaluating baseline model...")
    baseline_results = evaluate_model(args.baseline_model, args.n_eval_episodes)
    
    # Step 6: Generate evaluation comparison plots
    print("\n" + "="*80)
    print("GENERATING EVALUATION COMPARISON PLOTS")
    print("="*80)
    
    eval_plot, summary = plot_evaluation_comparison(
        optimized_results, baseline_results, args.output_dir
    )
    
    # Step 7: Generate final summary report
    print("\n" + "="*80)
    print("GENERATING FINAL REPORT")
    print("="*80)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'training_complete': training_complete,
        'final_model': final_model,
        'baseline_model': args.baseline_model,
        'evaluation_summary': summary,
        'plots': {
            'training_comparison': training_plot,
            'evaluation_comparison': eval_plot
        }
    }
    
    report_path = os.path.join(args.output_dir, f'final_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Final report saved to: {report_path}")
    
    print("\n" + "="*80)
    print("AUTO-COMPARE COMPLETE!")
    print("="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All results saved to: {args.output_dir}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
