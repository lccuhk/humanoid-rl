"""Feature Extractor for Humanoid RL"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque


class FeatureExtractor:
    """
    Feature extractor for humanoid robot reinforcement learning.
    
    This class extracts meaningful features from raw observation data,
    including:
    - Joint angle features
    - Velocity features
    - Contact detection features
    - Gait cycle features
    - Balance features
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        history_length: int = 4,
        use_velocity_features: bool = True,
        use_contact_features: bool = True,
        use_balance_features: bool = True,
        use_gait_features: bool = True,
    ):
        """
        Initialize feature extractor.
        
        Args:
            observation_dim: Dimension of raw observation
            action_dim: Dimension of action space
            history_length: Length of observation history to keep
            use_velocity_features: Whether to use velocity-based features
            use_contact_features: Whether to use contact detection features
            use_balance_features: Whether to use balance features
            use_gait_features: Whether to use gait cycle features
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.use_velocity_features = use_velocity_features
        self.use_contact_features = use_contact_features
        self.use_balance_features = use_balance_features
        self.use_gait_features = use_gait_features
        
        self.observation_history = deque(maxlen=history_length)
        self.action_history = deque(maxlen=history_length)
        self.step_count = 0
        
        self._feature_dim = self._calculate_feature_dim()
    
    def _calculate_feature_dim(self) -> int:
        """Calculate total feature dimension."""
        dim = self.observation_dim
        
        if self.use_velocity_features:
            dim += self.observation_dim
        
        if self.use_contact_features:
            dim += 4
        
        if self.use_balance_features:
            dim += 6
        
        if self.use_gait_features:
            dim += 8
        
        return dim
    
    def reset(self):
        """Reset the feature extractor state."""
        self.observation_history.clear()
        self.action_history.clear()
        self.step_count = 0
    
    def extract_velocity_features(
        self,
        current_obs: np.ndarray,
        previous_obs: np.ndarray
    ) -> np.ndarray:
        """
        Extract velocity-based features.
        
        Args:
            current_obs: Current observation
            previous_obs: Previous observation
            
        Returns:
            Velocity features
        """
        velocity = current_obs - previous_obs
        acceleration = np.zeros_like(velocity)
        
        if len(self.observation_history) >= 2:
            prev_prev_obs = self.observation_history[-2]
            prev_velocity = previous_obs - prev_prev_obs
            acceleration = velocity - prev_velocity
        
        return np.concatenate([velocity, acceleration])
    
    def extract_contact_features(
        self,
        observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract contact detection features.
        
        Args:
            observation: Current observation
            info: Additional information from environment
            
        Returns:
            Contact features
        """
        contact_features = np.zeros(4)
        
        if info is not None:
            if 'contact_forces' in info:
                contact_forces = np.array(info['contact_forces'])
                contact_features[0] = np.mean(contact_forces) > 0.1
                contact_features[1] = np.max(contact_forces)
                contact_features[2] = np.sum(contact_forces > 0.1)
                contact_features[3] = np.std(contact_forces)
        
        return contact_features
    
    def extract_balance_features(
        self,
        observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract balance-related features.
        
        Args:
            observation: Current observation
            info: Additional information from environment
            
        Returns:
            Balance features
        """
        balance_features = np.zeros(6)
        
        if info is not None:
            if 'z_position' in info:
                balance_features[0] = info['z_position']
            
            if 'x_velocity' in info:
                balance_features[1] = info['x_velocity']
            
            if 'is_healthy' in info:
                balance_features[2] = float(info['is_healthy'])
        
        if len(observation) > 3:
            balance_features[3] = observation[0] if len(observation) > 0 else 0
            balance_features[4] = observation[1] if len(observation) > 1 else 0
            balance_features[5] = observation[2] if len(observation) > 2 else 0
        
        return balance_features
    
    def extract_gait_features(
        self,
        observation: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        """
        Extract gait cycle features.
        
        Args:
            observation: Current observation
            action: Current action
            
        Returns:
            Gait features
        """
        gait_features = np.zeros(8)
        
        if len(self.action_history) > 0:
            action_diff = action - self.action_history[-1]
            gait_features[0] = np.mean(action_diff)
            gait_features[1] = np.std(action_diff)
            gait_features[2] = np.max(np.abs(action_diff))
        
        if len(self.observation_history) > 0:
            obs_diff = observation - self.observation_history[-1]
            gait_features[3] = np.mean(obs_diff)
            gait_features[4] = np.std(obs_diff)
        
        gait_features[5] = self.step_count % 100
        gait_features[6] = np.sin(2 * np.pi * self.step_count / 50)
        gait_features[7] = np.cos(2 * np.pi * self.step_count / 50)
        
        return gait_features
    
    def extract_features(
        self,
        observation: np.ndarray,
        action: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract all features from observation.
        
        Args:
            observation: Current observation
            action: Current action (optional)
            info: Additional information from environment (optional)
            
        Returns:
            Extracted feature vector
        """
        features = [observation.copy()]
        
        if self.use_velocity_features and len(self.observation_history) > 0:
            velocity_features = self.extract_velocity_features(
                observation, self.observation_history[-1]
            )
            features.append(velocity_features)
        
        if self.use_contact_features:
            contact_features = self.extract_contact_features(observation, info)
            features.append(contact_features)
        
        if self.use_balance_features:
            balance_features = self.extract_balance_features(observation, info)
            features.append(balance_features)
        
        if self.use_gait_features and action is not None:
            gait_features = self.extract_gait_features(observation, action)
            features.append(gait_features)
        
        self.observation_history.append(observation.copy())
        if action is not None:
            self.action_history.append(action.copy())
        self.step_count += 1
        
        return np.concatenate(features)
    
    def get_feature_dim(self) -> int:
        """
        Get the dimension of extracted features.
        
        Returns:
            Feature dimension
        """
        return self._feature_dim
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about extracted features.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'observation_dim': self.observation_dim,
            'action_dim': self.action_dim,
            'feature_dim': self._feature_dim,
            'history_length': self.history_length,
            'step_count': self.step_count,
            'use_velocity_features': self.use_velocity_features,
            'use_contact_features': self.use_contact_features,
            'use_balance_features': self.use_balance_features,
            'use_gait_features': self.use_gait_features
        }
