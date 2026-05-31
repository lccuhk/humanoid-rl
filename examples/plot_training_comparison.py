#!/usr/bin/env python3
"""
Plot training comparison between walk and run tasks.

This script generates comparison plots for:
- Episode rewards over time
- Reward components breakdown
- Forward velocity comparison
- Distance traveled comparison
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_training_data(csv_path: str) -> pd.DataFrame:
    """Load training data from CSV file."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def compute_episode_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute episode-level statistics from step-level data."""
    episode_stats = df.groupby('episode').agg({
        'reward': ['sum', 'mean', 'min', 'max', 'count'],
        'reward_forward': 'sum',
        'reward_position': 'sum',
        'reward_healthy': 'sum',
        'reward_ctrl': 'sum',
        'penalty_backward': 'sum',
        'x_velocity': 'mean',
        'distance_traveled': 'last',
        'z_position': 'mean'
    }).reset_index()
    
    episode_stats.columns = [
        'episode', 'total_reward', 'mean_reward_per_step', 'min_reward', 'max_reward',
        'episode_length', 'total_forward_reward', 'total_position_reward',
        'total_healthy_reward', 'total_ctrl_cost', 'total_backward_penalty',
        'mean_x_velocity', 'final_distance', 'mean_z_position'
    ]
    
    return episode_stats


def smooth_data(data: np.ndarray, window_size: int = 100) -> np.ndarray:
    """Smooth data using moving average."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_episode_rewards_comparison(
    walk_stats: pd.DataFrame,
    run_stats: pd.DataFrame,
    output_dir: str,
    window_size: int = 50
):
    """Plot episode rewards comparison between walk and run."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    ax1 = axes[0]
    ax1.plot(walk_stats['episode'], walk_stats['total_reward'], 
             alpha=0.3, color='blue', label='Walk (Raw)')
    ax1.plot(run_stats['episode'], run_stats['total_reward'], 
             alpha=0.3, color='red', label='Run (Raw)')
    
    if len(walk_stats) >= window_size:
        walk_smoothed = smooth_data(walk_stats['total_reward'].values, window_size)
        ax1.plot(walk_stats['episode'][window_size-1:], walk_smoothed, 
                 color='blue', linewidth=2, label=f'Walk (Smoothed, window={window_size})')
    
    if len(run_stats) >= window_size:
        run_smoothed = smooth_data(run_stats['total_reward'].values, window_size)
        ax1.plot(run_stats['episode'][window_size-1:], run_smoothed, 
                 color='red', linewidth=2, label=f'Run (Smoothed, window={window_size})')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Episode Rewards: Walk vs Run', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    walk_mean = walk_stats['total_reward'].rolling(window=window_size, min_periods=1).mean()
    run_mean = run_stats['total_reward'].rolling(window=window_size, min_periods=1).mean()
    walk_std = walk_stats['total_reward'].rolling(window=window_size, min_periods=1).std()
    run_std = run_stats['total_reward'].rolling(window=window_size, min_periods=1).std()
    
    ax2.plot(walk_stats['episode'], walk_mean, color='blue', linewidth=2, label='Walk (Rolling Mean)')
    ax2.fill_between(walk_stats['episode'], walk_mean - walk_std, walk_mean + walk_std, 
                     color='blue', alpha=0.2)
    ax2.plot(run_stats['episode'], run_mean, color='red', linewidth=2, label='Run (Rolling Mean)')
    ax2.fill_between(run_stats['episode'], run_mean - run_std, run_mean + run_std, 
                     color='red', alpha=0.2)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Total Reward', fontsize=12)
    ax2.set_title('Episode Rewards (Rolling Mean with Std Dev)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'episode_rewards_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_reward_components_comparison(
    walk_stats: pd.DataFrame,
    run_stats: pd.DataFrame,
    output_dir: str
):
    """Plot reward components comparison between walk and run."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    walk_components = {
        'Forward': walk_stats['total_forward_reward'].mean(),
        'Position': walk_stats['total_position_reward'].mean(),
        'Healthy': walk_stats['total_healthy_reward'].mean(),
        'Control Cost': walk_stats['total_ctrl_cost'].mean(),
        'Backward Penalty': walk_stats['total_backward_penalty'].mean()
    }
    
    run_components = {
        'Forward': run_stats['total_forward_reward'].mean(),
        'Position': run_stats['total_position_reward'].mean(),
        'Healthy': run_stats['total_healthy_reward'].mean(),
        'Control Cost': run_stats['total_ctrl_cost'].mean(),
        'Backward Penalty': run_stats['total_backward_penalty'].mean()
    }
    
    ax1 = axes[0]
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#F44336', '#9C27B0']
    x = np.arange(len(walk_components))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, list(walk_components.values()), width, 
                    label='Walk', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, list(run_components.values()), width, 
                    label='Run', color='red', alpha=0.7)
    
    ax1.set_xlabel('Reward Component', fontsize=12)
    ax1.set_ylabel('Mean Value per Episode', fontsize=12)
    ax1.set_title('Reward Components Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(walk_components.keys()), rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax2 = axes[1]
    walk_total = sum(abs(v) for v in walk_components.values())
    run_total = sum(abs(v) for v in run_components.values())
    
    walk_percentages = {k: abs(v) / walk_total * 100 for k, v in walk_components.items()}
    run_percentages = {k: abs(v) / run_total * 100 for k, v in run_components.items()}
    
    x = np.arange(len(walk_percentages))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, list(walk_percentages.values()), width, 
                    label='Walk', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, list(run_percentages.values()), width, 
                    label='Run', color='red', alpha=0.7)
    
    ax2.set_xlabel('Reward Component', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Reward Components Distribution (%)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(walk_percentages.keys()), rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'reward_components_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_velocity_comparison(
    walk_stats: pd.DataFrame,
    run_stats: pd.DataFrame,
    output_dir: str,
    window_size: int = 50
):
    """Plot forward velocity comparison between walk and run."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    ax1 = axes[0]
    ax1.plot(walk_stats['episode'], walk_stats['mean_x_velocity'], 
             alpha=0.3, color='blue', label='Walk (Raw)')
    ax1.plot(run_stats['episode'], run_stats['mean_x_velocity'], 
             alpha=0.3, color='red', label='Run (Raw)')
    
    if len(walk_stats) >= window_size:
        walk_smoothed = smooth_data(walk_stats['mean_x_velocity'].values, window_size)
        ax1.plot(walk_stats['episode'][window_size-1:], walk_smoothed, 
                 color='blue', linewidth=2, label=f'Walk (Smoothed)')
    
    if len(run_stats) >= window_size:
        run_smoothed = smooth_data(run_stats['mean_x_velocity'].values, window_size)
        ax1.plot(run_stats['episode'][window_size-1:], run_smoothed, 
                 color='red', linewidth=2, label=f'Run (Smoothed)')
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Mean Forward Velocity (m/s)', fontsize=12)
    ax1.set_title('Forward Velocity: Walk vs Run', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    walk_final = walk_stats['final_distance'].values
    run_final = run_stats['final_distance'].values
    
    ax2.plot(walk_stats['episode'], walk_final, 
             alpha=0.3, color='blue', label='Walk (Raw)')
    ax2.plot(run_stats['episode'], run_final, 
             alpha=0.3, color='red', label='Run (Raw)')
    
    if len(walk_stats) >= window_size:
        walk_smoothed = smooth_data(walk_final, window_size)
        ax2.plot(walk_stats['episode'][window_size-1:], walk_smoothed, 
                 color='blue', linewidth=2, label=f'Walk (Smoothed)')
    
    if len(run_stats) >= window_size:
        run_smoothed = smooth_data(run_final, window_size)
        ax2.plot(run_stats['episode'][window_size-1:], run_smoothed, 
                 color='red', linewidth=2, label=f'Run (Smoothed)')
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Final Distance (m)', fontsize=12)
    ax2.set_title('Distance Traveled: Walk vs Run', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'velocity_distance_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_episode_length_comparison(
    walk_stats: pd.DataFrame,
    run_stats: pd.DataFrame,
    output_dir: str,
    window_size: int = 50
):
    """Plot episode length comparison between walk and run."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(walk_stats['episode'], walk_stats['episode_length'], 
            alpha=0.3, color='blue', label='Walk (Raw)')
    ax.plot(run_stats['episode'], run_stats['episode_length'], 
            alpha=0.3, color='red', label='Run (Raw)')
    
    if len(walk_stats) >= window_size:
        walk_smoothed = smooth_data(walk_stats['episode_length'].values, window_size)
        ax.plot(walk_stats['episode'][window_size-1:], walk_smoothed, 
                color='blue', linewidth=2, label=f'Walk (Smoothed)')
    
    if len(run_stats) >= window_size:
        run_smoothed = smooth_data(run_stats['episode_length'].values, window_size)
        ax.plot(run_stats['episode'][window_size-1:], run_smoothed, 
                color='red', linewidth=2, label=f'Run (Smoothed)')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Length (steps)', fontsize=12)
    ax.set_title('Episode Length: Walk vs Run', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'episode_length_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_statistics(
    walk_stats: pd.DataFrame,
    run_stats: pd.DataFrame,
    output_dir: str
):
    """Plot summary statistics comparison."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    
    metrics = {
        'Mean Reward': (walk_stats['total_reward'].mean(), run_stats['total_reward'].mean()),
        'Max Reward': (walk_stats['total_reward'].max(), run_stats['total_reward'].max()),
        'Mean Episode Length': (walk_stats['episode_length'].mean(), run_stats['episode_length'].mean()),
        'Max Episode Length': (walk_stats['episode_length'].max(), run_stats['episode_length'].max()),
        'Mean Velocity': (walk_stats['mean_x_velocity'].mean(), run_stats['mean_x_velocity'].mean()),
        'Max Velocity': (walk_stats['mean_x_velocity'].max(), run_stats['mean_x_velocity'].max()),
        'Mean Distance': (walk_stats['final_distance'].mean(), run_stats['final_distance'].mean()),
        'Max Distance': (walk_stats['final_distance'].max(), run_stats['final_distance'].max())
    }
    
    for i, (metric_name, (walk_val, run_val)) in enumerate(metrics.items()):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        bars = ax.bar(['Walk', 'Run'], [walk_val, run_val], 
                      color=['blue', 'red'], alpha=0.7)
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'summary_statistics_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_comparison_summary(
    walk_stats: pd.DataFrame,
    run_stats: pd.DataFrame
):
    """Print comparison summary."""
    print("\n" + "="*70)
    print("WALK vs RUN - Training Comparison Summary")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Walk':>15} {'Run':>15} {'Difference':>15}")
    print("-"*70)
    
    metrics = [
        ('Total Episodes', len(walk_stats), len(run_stats)),
        ('Mean Reward', walk_stats['total_reward'].mean(), run_stats['total_reward'].mean()),
        ('Max Reward', walk_stats['total_reward'].max(), run_stats['total_reward'].max()),
        ('Mean Episode Length', walk_stats['episode_length'].mean(), run_stats['episode_length'].mean()),
        ('Max Episode Length', walk_stats['episode_length'].max(), run_stats['episode_length'].max()),
        ('Mean Forward Velocity', walk_stats['mean_x_velocity'].mean(), run_stats['mean_x_velocity'].mean()),
        ('Max Forward Velocity', walk_stats['mean_x_velocity'].max(), run_stats['mean_x_velocity'].max()),
        ('Mean Distance', walk_stats['final_distance'].mean(), run_stats['final_distance'].mean()),
        ('Max Distance', walk_stats['final_distance'].max(), run_stats['final_distance'].max()),
    ]
    
    for name, walk_val, run_val in metrics:
        diff = run_val - walk_val
        diff_pct = (diff / walk_val * 100) if walk_val != 0 else 0
        print(f"{name:<30} {walk_val:>15.2f} {run_val:>15.2f} {diff:>+15.2f} ({diff_pct:+.1f}%)")
    
    print("\n" + "="*70)


def main():
    """Main function to generate comparison plots."""
    parser = argparse.ArgumentParser(description='Plot training comparison between walk and run')
    parser.add_argument('--walk_csv', type=str, 
                        default='./logs/detailed/ppo_mujoco_walk_20260522_120133/step_logs_20260522_120135.csv',
                        help='Path to walk training CSV')
    parser.add_argument('--run_csv', type=str,
                        default='./logs/detailed/ppo_mujoco_run_20260531_170050/step_logs_20260531_170052.csv',
                        help='Path to run training CSV')
    parser.add_argument('--output_dir', type=str, default='./logs/comparison',
                        help='Output directory for plots')
    parser.add_argument('--window_size', type=int, default=50,
                        help='Window size for smoothing')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Training Comparison: Walk vs Run")
    print("="*70)
    
    print("\n1. Loading training data...")
    walk_df = load_training_data(args.walk_csv)
    run_df = load_training_data(args.run_csv)
    
    print("\n2. Computing episode statistics...")
    walk_stats = compute_episode_statistics(walk_df)
    run_stats = compute_episode_statistics(run_df)
    
    print(f"   Walk: {len(walk_stats)} episodes")
    print(f"   Run: {len(run_stats)} episodes")
    
    print("\n3. Generating comparison plots...")
    
    plot_episode_rewards_comparison(walk_stats, run_stats, args.output_dir, args.window_size)
    plot_reward_components_comparison(walk_stats, run_stats, args.output_dir)
    plot_velocity_comparison(walk_stats, run_stats, args.output_dir, args.window_size)
    plot_episode_length_comparison(walk_stats, run_stats, args.output_dir, args.window_size)
    plot_summary_statistics(walk_stats, run_stats, args.output_dir)
    
    print_comparison_summary(walk_stats, run_stats)
    
    print(f"\nAll plots saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
