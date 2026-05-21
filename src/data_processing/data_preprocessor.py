"""Data Preprocessor for Humanoid RL"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA


class DataPreprocessor:
    """
    Data preprocessor for humanoid robot reinforcement learning data.
    
    This class provides methods for:
    - Data normalization and standardization
    - Outlier detection and removal
    - Missing value handling
    - Feature scaling
    - Dimensionality reduction
    """
    
    def __init__(
        self,
        scaling_method: str = "standard",
        handle_outliers: bool = True,
        outlier_threshold: float = 3.0,
        pca_components: Optional[int] = None,
    ):
        """
        Initialize data preprocessor.
        
        Args:
            scaling_method: Scaling method ('standard', 'minmax', 'robust')
            handle_outliers: Whether to handle outliers
            outlier_threshold: Z-score threshold for outlier detection
            pca_components: Number of PCA components (None for no PCA)
        """
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        self.pca_components = pca_components
        
        self.scaler = None
        self.pca = None
        self.fitted = False
    
    def _get_scaler(self):
        """Get the appropriate scaler based on method."""
        if self.scaling_method == "standard":
            return StandardScaler()
        elif self.scaling_method == "minmax":
            return MinMaxScaler()
        elif self.scaling_method == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def fit(self, data: np.ndarray):
        """
        Fit the preprocessor to data.
        
        Args:
            data: Data to fit on
        """
        if self.handle_outliers:
            data = self._remove_outliers(data)
        
        self.scaler = self._get_scaler()
        self.scaler.fit(data)
        
        if self.pca_components is not None:
            scaled_data = self.scaler.transform(data)
            self.pca = PCA(n_components=self.pca_components)
            self.pca.fit(scaled_data)
        
        self.fitted = True
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        if self.handle_outliers:
            data = self._clip_outliers(data)
        
        scaled_data = self.scaler.transform(data)
        
        if self.pca is not None:
            scaled_data = self.pca.transform(scaled_data)
        
        return scaled_data
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform data.
        
        Args:
            data: Data to fit and transform
            
        Returns:
            Transformed data
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform data.
        
        Args:
            data: Data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        if self.pca is not None:
            data = self.pca.inverse_transform(data)
        
        return self.scaler.inverse_transform(data)
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        Remove outliers from data using Z-score method.
        
        Args:
            data: Input data
            
        Returns:
            Data with outliers removed
        """
        z_scores = np.abs((data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8))
        mask = (z_scores < self.outlier_threshold).all(axis=1)
        return data[mask]
    
    def _clip_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        Clip outliers to threshold values.
        
        Args:
            data: Input data
            
        Returns:
            Data with outliers clipped
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        lower = mean - self.outlier_threshold * std
        upper = mean + self.outlier_threshold * std
        return np.clip(data, lower, upper)
    
    def normalize_states(self, states: np.ndarray) -> np.ndarray:
        """
        Normalize state vectors.
        
        Args:
            states: State vectors
            
        Returns:
            Normalized states
        """
        if not self.fitted:
            self.fit(states)
        return self.transform(states)
    
    def normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
        Normalize reward values.
        
        Args:
            rewards: Reward values
            
        Returns:
            Normalized rewards
        """
        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-8
        return (rewards - mean) / std
    
    def compute_advantage(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        dones: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Reward values
            values: Value function estimates
            next_values: Next state value estimates
            dones: Done flags
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Advantage estimates
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values[t] * mask - values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * mask * last_advantage
        
        return advantages
    
    def compute_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        gamma: float = 0.99
    ) -> np.ndarray:
        """
        Compute discounted returns.
        
        Args:
            rewards: Reward values
            dones: Done flags
            gamma: Discount factor
            
        Returns:
            Discounted returns
        """
        returns = np.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            running_return = rewards[t] + gamma * running_return * mask
            returns[t] = running_return
        
        return returns
    
    def preprocess_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        values: Optional[np.ndarray] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess a complete trajectory.
        
        Args:
            states: State vectors
            actions: Action vectors
            rewards: Reward values
            next_states: Next state vectors
            dones: Done flags
            values: Value function estimates (optional)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Dictionary with preprocessed data
        """
        normalized_states = self.normalize_states(states)
        normalized_next_states = self.normalize_states(next_states)
        normalized_rewards = self.normalize_rewards(rewards)
        
        if values is not None:
            next_values = values[1:] if len(values) > len(rewards) else values
            advantages = self.compute_advantage(
                rewards, values[:-1], next_values, dones, gamma, gae_lambda
            )
        else:
            advantages = None
        
        returns = self.compute_returns(rewards, dones, gamma)
        
        return {
            'states': normalized_states,
            'actions': actions,
            'rewards': normalized_rewards,
            'next_states': normalized_next_states,
            'dones': dones,
            'advantages': advantages,
            'returns': returns
        }
    
    def save(self, filepath: str):
        """
        Save preprocessor state to file.
        
        Args:
            filepath: Path to save file
        """
        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'pca': self.pca,
            'scaling_method': self.scaling_method,
            'handle_outliers': self.handle_outliers,
            'outlier_threshold': self.outlier_threshold,
            'pca_components': self.pca_components,
            'fitted': self.fitted
        }, filepath)
    
    def load(self, filepath: str):
        """
        Load preprocessor state from file.
        
        Args:
            filepath: Path to load file
        """
        import joblib
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.pca = data['pca']
        self.scaling_method = data['scaling_method']
        self.handle_outliers = data['handle_outliers']
        self.outlier_threshold = data['outlier_threshold']
        self.pca_components = data['pca_components']
        self.fitted = data['fitted']
