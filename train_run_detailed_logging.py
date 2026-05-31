#!/usr/bin/env python3
"""
Training script for humanoid running task with detailed joint logging.

This script logs detailed joint angles, velocities, and forces at each step
for debugging and analysis purposes.
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environments import HumanoidMuJoCoEnv
from src.agents import PPOAgent
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import csv
import numpy as np
import mujoco


class DetailedJointLoggingCallback(BaseCallback):
    """Custom callback for detailed joint logging during training."""
    
    def __init__(self, log_dir, verbose=0, log_joint_info=True):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_joint_info = log_joint_info
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'training_log_detailed_{timestamp}.csv')
        self.joint_log_file = os.path.join(log_dir, f'joint_log_{timestamp}.csv')
        
        self.csv_writer = None
        self.joint_csv_writer = None
        self.total_episodes = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.joint_names = []
        self.actuator_names = []
        
    def _init_callback(self):
        """Initialize callback."""
        if self.csv_writer is None:
            self.csv_file = open(self.log_file, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Basic headers
            basic_headers = [
                'step', 'episode', 'reward', 'reward_forward', 'reward_position',
                'reward_velocity_bonus', 'reward_air_time', 'reward_energy_efficiency',
                'reward_healthy', 'reward_ctrl', 'reward_contact', 'penalty_backward',
                'penalty_fall', 'x_velocity', 'y_velocity', 'z_velocity',
                'x_position', 'y_position', 'z_position', 'distance_traveled',
                'air_time', 'foot_contact', 'is_healthy', 'timestamp'
            ]
            
            self.csv_writer.writerow(basic_headers)
        
        if self.log_joint_info and self.joint_csv_writer is None:
            self.joint_csv_file = open(self.joint_log_file, 'w', newline='')
            self.joint_csv_writer = csv.writer(self.joint_csv_file)
            
            # Get joint names from environment
            env = self.training_env.envs[0].unwrapped
            if hasattr(env, 'env') and env.env is not None:
                mj_model = env.env.unwrapped.model
                self.joint_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
                self.actuator_names = [mj_model.actuator(i).name for i in range(mj_model.nu)]
            elif hasattr(env, 'model'):
                mj_model = env.model
                self.joint_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
                self.actuator_names = [mj_model.actuator(i).name for i in range(mj_model.nu)]
            
            # Create joint headers
            joint_headers = ['step', 'episode', 'timestamp']
            
            # Joint positions (angles)
            for name in self.joint_names:
                joint_headers.append(f'joint_pos_{name}')
            
            # Joint velocities
            for name in self.joint_names:
                joint_headers.append(f'joint_vel_{name}')
            
            # Actuator controls (torques/forces)
            for name in self.actuator_names:
                joint_headers.append(f'actuator_ctrl_{name}')
            
            # Actuator forces
            for name in self.actuator_names:
                joint_headers.append(f'actuator_force_{name}')
            
            # Contact forces
            joint_headers.extend([
                'contact_count',
                'total_contact_force_x',
                'total_contact_force_y',
                'total_contact_force_z',
                'left_foot_force',
                'right_foot_force',
            ])
            
            self.joint_csv_writer.writerow(joint_headers)
            
            print(f"Logging {len(self.joint_names)} joints and {len(self.actuator_names)} actuators")
    
    def _get_joint_data(self, env):
        """Extract joint data from environment."""
        joint_data = {}
        
        try:
            if hasattr(env, 'env') and env.env is not None:
                mj_data = env.env.unwrapped.data
                mj_model = env.env.unwrapped.model
            elif hasattr(env, 'data'):
                mj_data = env.data
                mj_model = env.model
            else:
                return joint_data
            
            # Joint positions and velocities
            for i, name in enumerate(self.joint_names):
                joint_id = mj_model.joint(name).id
                qpos_addr = mj_model.jnt_qposadr[joint_id]
                qvel_addr = mj_model.jnt_dofadr[joint_id]
                
                if qpos_addr < len(mj_data.qpos):
                    joint_data[f'joint_pos_{name}'] = mj_data.qpos[qpos_addr]
                if qvel_addr < len(mj_data.qvel):
                    joint_data[f'joint_vel_{name}'] = mj_data.qvel[qvel_addr]
            
            # Actuator controls and forces
            for i, name in enumerate(self.actuator_names):
                if i < len(mj_data.ctrl):
                    joint_data[f'actuator_ctrl_{name}'] = mj_data.ctrl[i]
                if i < len(mj_data.actuator_force):
                    joint_data[f'actuator_force_{name}'] = mj_data.actuator_force[i]
            
            # Contact forces
            contact_count = mj_data.ncon
            total_force = np.zeros(3)
            left_foot_force = 0.0
            right_foot_force = 0.0
            
            for i in range(contact_count):
                contact = mj_data.contact[i]
                force = np.zeros(6)
                mujoco.mj_contactForce(mj_model, mj_data, i, force)
                
                total_force += force[:3]
                
                geom1_name = mj_model.geom(contact.geom1).name.lower()
                geom2_name = mj_model.geom(contact.geom2).name.lower()
                
                if 'left' in geom1_name or 'left' in geom2_name:
                    left_foot_force += np.linalg.norm(force[:3])
                if 'right' in geom1_name or 'right' in geom2_name:
                    right_foot_force += np.linalg.norm(force[:3])
            
            joint_data['contact_count'] = contact_count
            joint_data['total_contact_force_x'] = total_force[0]
            joint_data['total_contact_force_y'] = total_force[1]
            joint_data['total_contact_force_z'] = total_force[2]
            joint_data['left_foot_force'] = left_foot_force
            joint_data['right_foot_force'] = right_foot_force
            
        except Exception as e:
            print(f"Warning: Could not extract joint data: {e}")
        
        return joint_data
    
    def _on_step(self):
        """Called after each step."""
        reward = self.locals.get('rewards', [0])[0]
        info = self.locals.get('infos', [{}])[0]
        done = self.locals.get('dones', [False])[0]
        
        self.current_episode_reward += float(reward)
        
        # Basic logging
        step_log = {
            'step': self.num_timesteps,
            'episode': self.total_episodes + 1,
            'reward': float(reward),
            'reward_forward': float(info.get('reward_forward', 0)),
            'reward_position': float(info.get('reward_position', 0)),
            'reward_velocity_bonus': float(info.get('reward_velocity_bonus', 0)),
            'reward_air_time': float(info.get('reward_air_time', 0)),
            'reward_energy_efficiency': float(info.get('reward_energy_efficiency', 0)),
            'reward_healthy': float(info.get('reward_healthy', 0)),
            'reward_ctrl': float(info.get('reward_ctrl', 0)),
            'reward_contact': float(info.get('reward_contact', 0)),
            'penalty_backward': float(info.get('penalty_backward', 0)),
            'penalty_fall': float(info.get('penalty_fall', 0)),
            'x_velocity': float(info.get('x_velocity', 0)),
            'y_velocity': float(info.get('y_velocity', 0)),
            'z_velocity': float(info.get('z_velocity', 0)),
            'x_position': float(info.get('x_position', 0)),
            'y_position': float(info.get('y_position', 0)),
            'z_position': float(info.get('z_position', 0)),
            'distance_traveled': float(info.get('distance_traveled', 0)),
            'air_time': float(info.get('air_time', 0)),
            'foot_contact': int(info.get('foot_contact', True)),
            'is_healthy': int(info.get('is_healthy', True)),
            'timestamp': datetime.now().isoformat()
        }
        
        self.csv_writer.writerow([
            step_log['step'], step_log['episode'], step_log['reward'],
            step_log['reward_forward'], step_log['reward_position'],
            step_log['reward_velocity_bonus'], step_log['reward_air_time'],
            step_log['reward_energy_efficiency'], step_log['reward_healthy'],
            step_log['reward_ctrl'], step_log['reward_contact'],
            step_log['penalty_backward'], step_log['penalty_fall'],
            step_log['x_velocity'], step_log['y_velocity'], step_log['z_velocity'],
            step_log['x_position'], step_log['y_position'], step_log['z_position'],
            step_log['distance_traveled'], step_log['air_time'],
            step_log['foot_contact'], step_log['is_healthy'], step_log['timestamp']
        ])
        
        # Detailed joint logging
        if self.log_joint_info and self.joint_csv_writer is not None:
            env = self.training_env.envs[0].unwrapped
            joint_data = self._get_joint_data(env)
            
            if joint_data:
                joint_row = [
                    step_log['step'],
                    step_log['episode'],
                    step_log['timestamp']
                ]
                
                # Joint positions
                for name in self.joint_names:
                    joint_row.append(joint_data.get(f'joint_pos_{name}', 0.0))
                
                # Joint velocities
                for name in self.joint_names:
                    joint_row.append(joint_data.get(f'joint_vel_{name}', 0.0))
                
                # Actuator controls
                for name in self.actuator_names:
                    joint_row.append(joint_data.get(f'actuator_ctrl_{name}', 0.0))
                
                # Actuator forces
                for name in self.actuator_names:
                    joint_row.append(joint_data.get(f'actuator_force_{name}', 0.0))
                
                # Contact forces
                joint_row.extend([
                    joint_data.get('contact_count', 0),
                    joint_data.get('total_contact_force_x', 0.0),
                    joint_data.get('total_contact_force_y', 0.0),
                    joint_data.get('total_contact_force_z', 0.0),
                    joint_data.get('left_foot_force', 0.0),
                    joint_data.get('right_foot_force', 0.0),
                ])
                
                self.joint_csv_writer.writerow(joint_row)
        
        if done:
            self.total_episodes += 1
            self.episode_rewards.append(self.current_episode_reward)
            
            if self.total_episodes % 100 == 0:
                avg_reward = sum(self.episode_rewards[-100:]) / min(100, len(self.episode_rewards))
                print(f"Episode {self.total_episodes} | Avg Reward (last 100): {avg_reward:.2f} | "
                      f"Steps: {self.num_timesteps}")
            
            self.current_episode_reward = 0
        
        return True
    
    def _on_training_end(self):
        """Called at the end of training."""
        if self.csv_file:
            self.csv_file.close()
        if self.joint_csv_file:
            self.joint_csv_file.close()
        
        print(f"\nDetailed training log saved to: {self.log_file}")
        if self.log_joint_info:
            print(f"Joint log saved to: {self.joint_log_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train humanoid running policy with detailed joint logging'
    )
    
    parser.add_argument(
        '--total_timesteps', type=int, default=2000000,
        help='Total training timesteps (default: 2,000,000)'
    )
    
    parser.add_argument(
        '--learning_rate', type=float, default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    
    parser.add_argument(
        '--n_steps', type=int, default=2048,
        help='Number of steps per update (default: 2048)'
    )
    
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size (default: 64)'
    )
    
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='Discount factor (default: 0.99)'
    )
    
    parser.add_argument(
        '--gae_lambda', type=float, default=0.95,
        help='GAE lambda (default: 0.95)'
    )
    
    parser.add_argument(
        '--clip_range', type=float, default=0.2,
        help='PPO clip range (default: 0.2)'
    )
    
    parser.add_argument(
        '--healthy_reward', type=float, default=3.0,
        help='Healthy reward weight (default: 3.0)'
    )
    
    parser.add_argument(
        '--forward_reward_weight', type=float, default=5.0,
        help='Forward reward weight (default: 5.0)'
    )
    
    parser.add_argument(
        '--backward_penalty_weight', type=float, default=10.0,
        help='Backward penalty weight (default: 10.0)'
    )
    
    parser.add_argument(
        '--position_reward_weight', type=float, default=2.0,
        help='Position reward weight (default: 2.0)'
    )
    
    parser.add_argument(
        '--velocity_bonus_weight', type=float, default=30.0,
        help='Weight for velocity bonus reward (default: 30.0)'
    )
    
    parser.add_argument(
        '--air_time_reward_weight', type=float, default=50.0,
        help='Weight for air time reward (default: 50.0)'
    )
    
    parser.add_argument(
        '--energy_efficiency_weight', type=float, default=5.0,
        help='Weight for energy efficiency reward (default: 5.0)'
    )
    
    parser.add_argument(
        '--fall_penalty', type=float, default=100.0,
        help='Fall penalty (default: 100.0)'
    )
    
    parser.add_argument(
        '--no_joint_logging', action='store_true',
        default=False,
        help='Disable detailed joint logging (default: False)'
    )
    
    parser.add_argument(
        '--checkpoint_dir', type=str, default='./checkpoints',
        help='Directory to save checkpoints (default: ./checkpoints)'
    )
    
    parser.add_argument(
        '--log_dir', type=str, default='./logs',
        help='Directory to save logs (default: ./logs)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_mujoco_run_detailed_{timestamp}"
    
    print("=" * 80)
    print("Humanoid Running Training with Detailed Joint Logging")
    print("=" * 80)
    print(f"Run name: {run_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Joint logging: {not args.no_joint_logging}")
    print()
    print("Reward Weights:")
    print(f"  Healthy reward: {args.healthy_reward}")
    print(f"  Forward reward: {args.forward_reward_weight}")
    print(f"  Position reward: {args.position_reward_weight}")
    print(f"  Velocity bonus: {args.velocity_bonus_weight}")
    print(f"  Air time reward: {args.air_time_reward_weight}")
    print(f"  Energy efficiency: {args.energy_efficiency_weight}")
    print(f"  Backward penalty: {args.backward_penalty_weight}")
    print(f"  Fall penalty: {args.fall_penalty}")
    print("=" * 80)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("\nCreating environment...")
    env = HumanoidMuJoCoEnv(
        render_mode=None,
        task='run',
        forward_reward_weight=args.forward_reward_weight,
        healthy_reward=args.healthy_reward,
        backward_penalty_weight=args.backward_penalty_weight,
        position_reward_weight=args.position_reward_weight,
        velocity_bonus_weight=args.velocity_bonus_weight,
        air_time_reward_weight=args.air_time_reward_weight,
        energy_efficiency_weight=args.energy_efficiency_weight,
        fall_penalty=args.fall_penalty,
    )
    
    print("Creating agent...")
    agent = PPOAgent(
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        verbose=1,
        tensorboard_log=os.path.join(args.log_dir, 'tensorboard', run_name),
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=args.checkpoint_dir,
        name_prefix=run_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    logging_callback = DetailedJointLoggingCallback(
        log_dir=os.path.join(args.log_dir, 'training', run_name),
        verbose=1,
        log_joint_info=not args.no_joint_logging,
    )
    
    print("\nStarting training...")
    print("=" * 80)
    
    try:
        agent.train(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, logging_callback],
            tb_log_name=run_name,
            reset_num_timesteps=True,
            progress_bar=True,
        )
        
        final_model_path = os.path.join(args.checkpoint_dir, f"{run_name}_final.zip")
        agent.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        final_model_path = os.path.join(args.checkpoint_dir, f"{run_name}_interrupted.zip")
        agent.save(final_model_path)
        print(f"Model saved to: {final_model_path}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    
    env.close()


if __name__ == "__main__":
    main()
