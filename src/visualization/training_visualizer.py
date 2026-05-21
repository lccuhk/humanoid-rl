"""Training Visualizer for Humanoid RL"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from typing import Dict, Any, List, Optional
from collections import defaultdict
import json

style.use('seaborn-v0_8-darkgrid')


class TrainingVisualizer:
    """
    Visualizer for training metrics and progress.
    
    This class provides visualization for:
    - Training rewards
    - Episode lengths
    - Loss curves
    - Learning rate schedules
    - Comparison between different algorithms
    """
    
    def __init__(
        self,
        output_dir: str = "./logs/visualizations",
        figsize: tuple = (12, 8),
        dpi: int = 100,
    ):
        """
        Initialize training visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            figsize: Figure size
            dpi: DPI for saved figures
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = defaultdict(list)
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """
        Log a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Step number (optional)
        """
        self.metrics[metric_name].append({
            'value': value,
            'step': step if step is not None else len(self.metrics[metric_name])
        })
    
    def log_episode(self, reward: float, length: int):
        """
        Log episode data.
        
        Args:
            reward: Episode reward
            length: Episode length
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
    
    def log_loss(self, loss_name: str, value: float):
        """
        Log loss value.
        
        Args:
            loss_name: Name of the loss
            value: Loss value
        """
        self.losses[loss_name].append(value)
    
    def plot_rewards(
        self,
        window_size: int = 100,
        show: bool = False,
        save: bool = True,
        filename: str = "rewards.png"
    ):
        """
        Plot episode rewards with moving average.
        
        Args:
            window_size: Window size for moving average
            show: Whether to show the plot
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.episode_rewards:
            print("No reward data to plot")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        episodes = np.arange(len(self.episode_rewards))
        rewards = np.array(self.episode_rewards)
        
        ax.plot(episodes, rewards, alpha=0.3, label='Raw Rewards', color='lightblue')
        
        if len(rewards) >= window_size:
            moving_avg = np.convolve(
                rewards,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            ax.plot(
                episodes[window_size-1:],
                moving_avg,
                label=f'Moving Avg (window={window_size})',
                color='blue',
                linewidth=2
            )
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Training Rewards', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved rewards plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_episode_lengths(
        self,
        window_size: int = 100,
        show: bool = False,
        save: bool = True,
        filename: str = "episode_lengths.png"
    ):
        """
        Plot episode lengths.
        
        Args:
            window_size: Window size for moving average
            show: Whether to show the plot
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.episode_lengths:
            print("No episode length data to plot")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        episodes = np.arange(len(self.episode_lengths))
        lengths = np.array(self.episode_lengths)
        
        ax.plot(episodes, lengths, alpha=0.3, label='Raw Lengths', color='lightgreen')
        
        if len(lengths) >= window_size:
            moving_avg = np.convolve(
                lengths,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            ax.plot(
                episodes[window_size-1:],
                moving_avg,
                label=f'Moving Avg (window={window_size})',
                color='green',
                linewidth=2
            )
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Episode Length', fontsize=12)
        ax.set_title('Episode Lengths Over Training', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved episode lengths plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_losses(
        self,
        show: bool = False,
        save: bool = True,
        filename: str = "losses.png"
    ):
        """
        Plot loss curves.
        
        Args:
            show: Whether to show the plot
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.losses:
            print("No loss data to plot")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.losses)))
        
        for i, (loss_name, values) in enumerate(self.losses.items()):
            steps = np.arange(len(values))
            ax.plot(steps, values, label=loss_name, color=colors[i], linewidth=1.5)
        
        ax.set_xlabel('Update Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Losses', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved losses plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_metrics(
        self,
        metric_names: Optional[List[str]] = None,
        show: bool = False,
        save: bool = True,
        filename: str = "metrics.png"
    ):
        """
        Plot multiple metrics.
        
        Args:
            metric_names: List of metric names to plot (None for all)
            show: Whether to show the plot
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.metrics:
            print("No metric data to plot")
            return
        
        if metric_names is None:
            metric_names = list(self.metrics.keys())
        
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(self.figsize[0], self.figsize[1] * n_metrics / 2), dpi=self.dpi)
        
        if n_metrics == 1:
            axes = [axes]
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))
        
        for i, (ax, metric_name) in enumerate(zip(axes, metric_names)):
            if metric_name not in self.metrics:
                continue
            
            data = self.metrics[metric_name]
            steps = [d['step'] for d in data]
            values = [d['value'] for d in data]
            
            ax.plot(steps, values, label=metric_name, color=colors[i], linewidth=1.5)
            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_title(f'{metric_name} Over Training', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved metrics plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        metric: str = 'rewards',
        show: bool = False,
        save: bool = True,
        filename: str = "comparison.png"
    ):
        """
        Plot comparison between different algorithms or runs.
        
        Args:
            results_dict: Dictionary mapping run names to results
            metric: Metric to compare ('rewards' or custom)
            show: Whether to show the plot
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
        
        for i, (name, results) in enumerate(results_dict.items()):
            if metric in results:
                values = results[metric]
                episodes = np.arange(len(values))
                
                ax.plot(episodes, values, label=name, color=colors[i], linewidth=1.5, alpha=0.8)
                
                if len(values) >= 50:
                    moving_avg = np.convolve(values, np.ones(50) / 50, mode='valid')
                    ax.plot(
                        episodes[49:],
                        moving_avg,
                        color=colors[i],
                        linewidth=2.5,
                        linestyle='--'
                    )
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'Algorithm Comparison - {metric.capitalize()}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved comparison plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_training_dashboard(
        self,
        show: bool = False,
        save: bool = True,
        filename: str = "training_dashboard.png"
    ):
        """
        Create a comprehensive training dashboard.
        
        Args:
            show: Whether to show the plot
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, :])
        
        if self.episode_rewards:
            episodes = np.arange(len(self.episode_rewards))
            rewards = np.array(self.episode_rewards)
            
            ax1.plot(episodes, rewards, alpha=0.3, color='lightblue')
            if len(rewards) >= 50:
                moving_avg = np.convolve(rewards, np.ones(50) / 50, mode='valid')
                ax1.plot(episodes[49:], moving_avg, color='blue', linewidth=2)
            ax1.set_xlabel('Episode', fontsize=10)
            ax1.set_ylabel('Reward', fontsize=10)
            ax1.set_title('Episode Rewards', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        if self.episode_lengths:
            episodes = np.arange(len(self.episode_lengths))
            lengths = np.array(self.episode_lengths)
            
            ax2.plot(episodes, lengths, alpha=0.3, color='lightgreen')
            if len(lengths) >= 50:
                moving_avg = np.convolve(lengths, np.ones(50) / 50, mode='valid')
                ax2.plot(episodes[49:], moving_avg, color='green', linewidth=2)
            ax2.set_xlabel('Episode', fontsize=10)
            ax2.set_ylabel('Length', fontsize=10)
            ax2.set_title('Episode Lengths', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        if self.episode_rewards:
            rewards = np.array(self.episode_rewards)
            ax3.hist(rewards, bins=30, color='orange', alpha=0.7, edgecolor='black')
            ax3.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
            ax3.set_xlabel('Reward', fontsize=10)
            ax3.set_ylabel('Frequency', fontsize=10)
            ax3.set_title('Reward Distribution', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
        
        if self.losses:
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.losses)))
            for i, (loss_name, values) in enumerate(self.losses.items()):
                ax4.plot(values, label=loss_name, color=colors[i], linewidth=1.5)
            ax4.set_xlabel('Update Step', fontsize=10)
            ax4.set_ylabel('Loss', fontsize=10)
            ax4.set_title('Training Losses', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
        
        fig.suptitle('Training Dashboard', fontsize=16, fontweight='bold')
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved training dashboard to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_data(self, filename: str = "visualization_data.json"):
        """
        Save visualization data to JSON file.
        
        Args:
            filename: Filename for saved data
        """
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': dict(self.losses),
            'metrics': dict(self.metrics)
        }
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved visualization data to {filepath}")
    
    def load_data(self, filepath: str):
        """
        Load visualization data from JSON file.
        
        Args:
            filepath: Path to data file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_lengths = data.get('episode_lengths', [])
        self.losses = defaultdict(list, data.get('losses', {}))
        self.metrics = defaultdict(list, data.get('metrics', {}))
        
        print(f"Loaded visualization data from {filepath}")
