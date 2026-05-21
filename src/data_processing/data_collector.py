"""Data Collector for Humanoid RL Training"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict


class DataCollector:
    """
    Data collector for humanoid robot reinforcement learning.
    
    This class collects, stores, and manages training data including:
    - Episode rewards and lengths
    - State transitions
    - Action distributions
    - Environment metrics
    """
    
    def __init__(
        self,
        output_dir: str = "./data",
        save_interval: int = 100,
        max_buffer_size: int = 10000,
    ):
        """
        Initialize data collector.
        
        Args:
            output_dir: Directory to save collected data
            save_interval: How often to save data to disk
            max_buffer_size: Maximum size of in-memory buffer
        """
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.max_buffer_size = max_buffer_size
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.episode_buffer = []
        self.step_buffer = []
        self.episode_count = 0
        self.step_count = 0
        
        self.training_metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        
        self.start_time = datetime.now()
    
    def collect_episode(
        self,
        episode_data: Dict[str, Any]
    ):
        """
        Collect episode-level data.
        
        Args:
            episode_data: Dictionary containing episode data
        """
        self.episode_buffer.append({
            'episode_id': self.episode_count,
            'timestamp': datetime.now().isoformat(),
            **episode_data
        })
        
        for key, value in episode_data.items():
            if isinstance(value, (int, float, np.number)):
                self.episode_metrics[key].append(value)
        
        self.episode_count += 1
        
        if self.episode_count % self.save_interval == 0:
            self.save_episodes()
    
    def collect_step(
        self,
        step_data: Dict[str, Any]
    ):
        """
        Collect step-level data.
        
        Args:
            step_data: Dictionary containing step data
        """
        self.step_buffer.append({
            'step_id': self.step_count,
            'episode_id': self.episode_count,
            'timestamp': datetime.now().isoformat(),
            **step_data
        })
        
        for key, value in step_data.items():
            if isinstance(value, (int, float, np.number)):
                self.training_metrics[key].append(value)
        
        self.step_count += 1
        
        if len(self.step_buffer) >= self.max_buffer_size:
            self.save_steps()
    
    def collect_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ):
        """
        Collect a transition tuple (s, a, r, s', done).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            info: Additional information
        """
        transition_data = {
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': action.tolist() if isinstance(action, np.ndarray) else action,
            'reward': reward,
            'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            'done': done
        }
        
        if info:
            transition_data['info'] = info
        
        self.collect_step(transition_data)
    
    def save_episodes(self, filename: Optional[str] = None):
        """
        Save episode buffer to disk.
        
        Args:
            filename: Optional custom filename
        """
        if not self.episode_buffer:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"episodes_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.episode_buffer, f, indent=2)
        
        print(f"Saved {len(self.episode_buffer)} episodes to {filepath}")
        self.episode_buffer = []
    
    def save_steps(self, filename: Optional[str] = None):
        """
        Save step buffer to disk.
        
        Args:
            filename: Optional custom filename
        """
        if not self.step_buffer:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"steps_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.step_buffer, f, indent=2)
        
        print(f"Saved {len(self.step_buffer)} steps to {filepath}")
        self.step_buffer = []
    
    def save_metrics(self, filename: Optional[str] = None):
        """
        Save all collected metrics to disk.
        
        Args:
            filename: Optional custom filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        metrics_data = {
            'metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_episodes': self.episode_count,
                'total_steps': self.step_count
            },
            'episode_metrics': dict(self.episode_metrics),
            'training_metrics': dict(self.training_metrics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"Saved metrics to {filepath}")
    
    def to_dataframe(self, data_type: str = 'episodes') -> pd.DataFrame:
        """
        Convert collected data to pandas DataFrame.
        
        Args:
            data_type: Type of data to convert ('episodes' or 'steps')
            
        Returns:
            DataFrame containing the data
        """
        if data_type == 'episodes':
            return pd.DataFrame(self.episode_buffer)
        elif data_type == 'steps':
            return pd.DataFrame(self.step_buffer)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics of collected data.
        
        Returns:
            Dictionary containing statistics
        """
        stats = {
            'total_episodes': self.episode_count,
            'total_steps': self.step_count,
            'duration_seconds': (datetime.now() - self.start_time).total_seconds()
        }
        
        for key, values in self.episode_metrics.items():
            if values:
                stats[f'episode_{key}_mean'] = np.mean(values)
                stats[f'episode_{key}_std'] = np.std(values)
                stats[f'episode_{key}_min'] = np.min(values)
                stats[f'episode_{key}_max'] = np.max(values)
        
        for key, values in self.training_metrics.items():
            if values:
                stats[f'step_{key}_mean'] = np.mean(values)
                stats[f'step_{key}_std'] = np.std(values)
        
        return stats
    
    def flush(self):
        """Flush all buffers to disk."""
        self.save_episodes()
        self.save_steps()
        self.save_metrics()
    
    def reset(self):
        """Reset all buffers and counters."""
        self.flush()
        self.episode_buffer = []
        self.step_buffer = []
        self.episode_count = 0
        self.step_count = 0
        self.training_metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.start_time = datetime.now()
