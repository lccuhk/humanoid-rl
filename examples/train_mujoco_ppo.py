#!/usr/bin/env python3
"""
Example: Train Humanoid with MuJoCo + PPO

This script demonstrates how to train a humanoid robot to walk
using MuJoCo physics engine and PPO algorithm.
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from src.environments import HumanoidMuJoCoEnv
from src.agents import PPOAgent
from src.data_processing import DataCollector
from src.visualization import TrainingVisualizer


def main():
    print("="*70)
    print("Example: MuJoCo + PPO for Humanoid Walking")
    print("="*70)
    
    log_dir = "./logs/mujoco_ppo_walk"
    checkpoint_dir = "./checkpoints/mujoco_ppo_walk"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\n1. Creating MuJoCo environment...")
    env = HumanoidMuJoCoEnv(
        render_mode=None,
        task="walk",
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
    )
    
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    print("\n2. Creating PPO agent...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    agent = PPOAgent(
        env=env,
        policy='MlpPolicy',
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        verbose=1,
        seed=42,
    )
    
    print("\n3. Setting up data collection...")
    data_collector = DataCollector(
        output_dir=os.path.join(log_dir, "data"),
        save_interval=100,
    )
    
    training_visualizer = TrainingVisualizer(
        output_dir=os.path.join(log_dir, "visualizations"),
    )
    
    print("\n4. Starting training...")
    print("   This will train for 1,000,000 timesteps (about 1-2 hours on GPU)")
    print("   Press Ctrl+C to stop early and save progress")
    
    total_timesteps = 1_000_000
    
    try:
        agent.train(
            total_timesteps=total_timesteps,
            log_interval=1,
            tb_log_name=f"ppo_walk_{timestamp}",
            progress_bar=True,
        )
        
        print("\n5. Training completed!")
        
        model_path = os.path.join(checkpoint_dir, f"ppo_walk_final_{timestamp}.zip")
        agent.save(model_path)
        print(f"   Model saved to: {model_path}")
        
        print("\n6. Evaluating trained model...")
        eval_env = HumanoidMuJoCoEnv(
            render_mode=None,
            task="walk",
        )
        
        eval_results = agent.evaluate(eval_env, n_episodes=10, deterministic=True)
        
        print("\n   Evaluation Results:")
        for key, value in eval_results.items():
            print(f"   - {key}: {value:.4f}")
        
        print("\n7. Generating visualizations...")
        training_visualizer.create_training_dashboard(save=True)
        
        print("\n8. Saving training configuration...")
        config = {
            'environment': 'mujoco',
            'algorithm': 'ppo',
            'task': 'walk',
            'total_timesteps': total_timesteps,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'clip_range': 0.2,
            'seed': 42,
            'timestamp': timestamp,
            'model_path': model_path,
            'evaluation_results': eval_results,
        }
        
        config_path = os.path.join(log_dir, f"training_config_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"   Config saved to: {config_path}")
        
        eval_env.close()
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        model_path = os.path.join(checkpoint_dir, f"ppo_walk_interrupted_{timestamp}.zip")
        agent.save(model_path)
        print(f"   Model saved to: {model_path}")
    
    finally:
        env.close()
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    print("\nNext steps:")
    print("1. View TensorBoard logs: tensorboard --logdir ./logs/mujoco_ppo_walk/tensorboard")
    print("2. Run evaluation: python examples/evaluate_model.py")
    print("3. Visualize results: python examples/visualize_results.py")
    print("4. Start web UI: python examples/run_web_ui.py")


if __name__ == "__main__":
    main()
