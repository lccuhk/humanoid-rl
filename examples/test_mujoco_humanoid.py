#!/usr/bin/env python3
"""
Test MuJoCo Humanoid Environment

This script tests the MuJoCo humanoid environment to ensure it works correctly.
"""

import os
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mujoco_import():
    """Test MuJoCo import."""
    print("="*70)
    print("Step 1: Testing MuJoCo Import")
    print("="*70)
    
    try:
        import mujoco
        print(f"✅ MuJoCo version: {mujoco.__version__}")
        return True
    except ImportError as e:
        print(f"❌ MuJoCo import failed: {e}")
        return False


def test_gymnasium_humanoid():
    """Test Gymnasium Humanoid environment."""
    print("\n" + "="*70)
    print("Step 2: Testing Gymnasium Humanoid Environment")
    print("="*70)
    
    try:
        import gymnasium as gym
        
        print("Creating Humanoid-v5 environment...")
        env = gym.make("Humanoid-v5")
        
        print(f"✅ Observation space: {env.observation_space}")
        print(f"✅ Action space: {env.action_space}")
        print(f"✅ Observation dim: {env.observation_space.shape[0]}")
        print(f"✅ Action dim: {env.action_space.shape[0]}")
        
        print("\nTesting environment reset...")
        obs, info = env.reset()
        print(f"✅ Reset successful, observation shape: {obs.shape}")
        
        print("\nTesting environment step...")
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Step successful")
        print(f"   Reward: {reward:.4f}")
        print(f"   Terminated: {terminated}")
        print(f"   Truncated: {truncated}")
        
        env.close()
        print("\n✅ Gymnasium Humanoid environment works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Gymnasium Humanoid test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_humanoid_env():
    """Test custom HumanoidMuJoCoEnv."""
    print("\n" + "="*70)
    print("Step 3: Testing Custom HumanoidMuJoCoEnv")
    print("="*70)
    
    try:
        from src.environments import HumanoidMuJoCoEnv
        
        print("Creating custom HumanoidMuJoCoEnv (walk task)...")
        env = HumanoidMuJoCoEnv(
            render_mode=None,
            task="walk",
        )
        
        print(f"✅ Observation space: {env.observation_space}")
        print(f"✅ Action space: {env.action_space}")
        print(f"✅ Task: {env.task}")
        
        print("\nTesting environment reset...")
        obs, info = env.reset()
        print(f"✅ Reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Info keys: {list(info.keys())}")
        
        print("\nTesting multiple steps...")
        total_reward = 0
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if i < 3:
                print(f"   Step {i+1}: reward={reward:.4f}, terminated={terminated}")
        
        print(f"\n✅ 10 steps completed successfully")
        print(f"   Total reward: {total_reward:.4f}")
        
        print("\nTesting get_state()...")
        state = env.get_state()
        print(f"✅ State keys: {list(state.keys())}")
        print(f"   X position: {state.get('x_position', 'N/A'):.4f}")
        print(f"   Z position: {state.get('z_position', 'N/A'):.4f}")
        print(f"   Is healthy: {state.get('is_healthy', 'N/A')}")
        
        env.close()
        print("\n✅ Custom HumanoidMuJoCoEnv works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Custom HumanoidMuJoCoEnv test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_agent_with_humanoid():
    """Test PPO agent with humanoid environment."""
    print("\n" + "="*70)
    print("Step 4: Testing PPO Agent with Humanoid")
    print("="*70)
    
    try:
        from src.environments import HumanoidMuJoCoEnv
        from src.agents import PPOAgent
        
        print("Creating environment...")
        env = HumanoidMuJoCoEnv(render_mode=None, task="walk")
        
        print("Creating PPO agent...")
        agent = PPOAgent(
            env=env,
            policy='MlpPolicy',
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=3,
            gamma=0.99,
            verbose=1,
            seed=42,
        )
        
        print("\n✅ PPO agent created successfully!")
        print(f"   Policy network initialized")
        print(f"   Learning rate: {agent.learning_rate}")
        print(f"   Steps per update: {agent.n_steps}")
        
        print("\nRunning quick training (1000 steps)...")
        agent.train(
            total_timesteps=1000,
            log_interval=1,
            progress_bar=True,
        )
        
        print("\n✅ Quick training completed!")
        
        print("\nTesting prediction...")
        obs, _ = env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        print(f"✅ Prediction successful")
        print(f"   Action shape: {action.shape}")
        print(f"   Action range: [{action.min():.4f}, {action.max():.4f}]")
        
        env.close()
        print("\n✅ PPO agent works correctly with Humanoid environment!")
        return True
        
    except Exception as e:
        print(f"❌ PPO agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MuJoCo Humanoid Environment Test Suite")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    results['MuJoCo Import'] = test_mujoco_import()
    results['Gymnasium Humanoid'] = test_gymnasium_humanoid()
    results['Custom HumanoidMuJoCoEnv'] = test_custom_humanoid_env()
    results['PPO Agent with Humanoid'] = test_ppo_agent_with_humanoid()
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 All tests passed! MuJoCo Humanoid environment is ready!")
        print("\nNext steps:")
        print("1. Run full training: python train.py --env mujoco --algo ppo --task walk")
        print("2. Train for longer: --total_timesteps 1000000")
        print("3. Try running task: --task run")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
