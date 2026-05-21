"""Tests for humanoid robot environments."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMuJoCoEnv:
    """Tests for MuJoCo environment."""
    
    @pytest.fixture
    def env(self):
        """Create MuJoCo environment fixture."""
        from src.environments import HumanoidMuJoCoEnv
        env = HumanoidMuJoCoEnv(render_mode=None, task="walk")
        yield env
        env.close()
    
    def test_observation_space(self, env):
        """Test observation space."""
        assert env.observation_space is not None
        assert env.observation_space.shape is not None
        assert len(env.observation_space.shape) == 1
    
    def test_action_space(self, env):
        """Test action space."""
        assert env.action_space is not None
        assert env.action_space.shape is not None
        assert len(env.action_space.shape) == 1
    
    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
    
    def test_step(self, env):
        """Test environment step."""
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_get_state(self, env):
        """Test get_state method."""
        env.reset()
        state = env.get_state()
        assert isinstance(state, dict)
        assert 'x_position' in state
        assert 'z_position' in state
        assert 'is_healthy' in state


class TestPyBulletEnv:
    """Tests for PyBullet environment."""
    
    @pytest.fixture
    def env(self):
        """Create PyBullet environment fixture."""
        try:
            from src.environments import HumanoidPyBulletEnv
            env = HumanoidPyBulletEnv(render_mode=None, task="walk")
            yield env
            env.close()
        except ImportError:
            pytest.skip("PyBullet not installed")
    
    def test_observation_space(self, env):
        """Test observation space."""
        assert env.observation_space is not None
        assert env.observation_space.shape is not None
    
    def test_action_space(self, env):
        """Test action space."""
        assert env.action_space is not None
        assert env.action_space.shape is not None
    
    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
    
    def test_step(self, env):
        """Test environment step."""
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


class TestDataProcessing:
    """Tests for data processing modules."""
    
    def test_data_collector(self):
        """Test data collector."""
        from src.data_processing import DataCollector
        
        collector = DataCollector(output_dir="./test_data")
        
        collector.collect_episode({'reward': 100.0, 'length': 100})
        collector.collect_episode({'reward': 150.0, 'length': 150})
        
        stats = collector.get_statistics()
        assert stats['total_episodes'] == 2
        
        import shutil
        shutil.rmtree("./test_data", ignore_errors=True)
    
    def test_data_preprocessor(self):
        """Test data preprocessor."""
        from src.data_processing import DataPreprocessor
        
        preprocessor = DataPreprocessor(scaling_method="standard")
        
        data = np.random.randn(100, 10)
        scaled = preprocessor.fit_transform(data)
        
        assert scaled.shape == data.shape
        assert np.allclose(np.mean(scaled, axis=0), 0, atol=1e-6)
    
    def test_feature_extractor(self):
        """Test feature extractor."""
        from src.data_processing import FeatureExtractor
        
        extractor = FeatureExtractor(
            observation_dim=20,
            action_dim=10,
            history_length=4
        )
        
        obs = np.random.randn(20)
        action = np.random.randn(10)
        
        features = extractor.extract_features(obs, action)
        
        assert features is not None
        assert isinstance(features, np.ndarray)


class TestVisualization:
    """Tests for visualization modules."""
    
    def test_training_visualizer(self):
        """Test training visualizer."""
        from src.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(output_dir="./test_viz")
        
        for i in range(100):
            visualizer.log_episode(reward=float(np.random.randn() * 100 + 500), length=100)
        
        assert len(visualizer.episode_rewards) == 100
        
        import shutil
        shutil.rmtree("./test_viz", ignore_errors=True)
    
    def test_robot_visualizer(self):
        """Test robot visualizer."""
        from src.visualization import RobotVisualizer
        
        visualizer = RobotVisualizer(output_dir="./test_robot_viz")
        
        for i in range(50):
            visualizer.log_state({
                'x_position': float(i * 0.1),
                'y_position': 0.0,
                'z_position': 1.0 + np.sin(i * 0.1) * 0.1,
                'x_velocity': 0.5,
                'is_healthy': True
            })
        
        assert len(visualizer.state_history) == 50
        
        import shutil
        shutil.rmtree("./test_robot_viz", ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
