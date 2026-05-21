"""Robot Visualizer for Humanoid RL"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Dict, Any, List, Optional, Tuple
import json


class RobotVisualizer:
    """
    Visualizer for humanoid robot states and animations.
    
    This class provides visualization for:
    - Robot joint positions
    - Robot trajectories
    - Gait analysis
    - 3D robot visualization
    """
    
    def __init__(
        self,
        output_dir: str = "./logs/visualizations",
        figsize: tuple = (12, 8),
        dpi: int = 100,
    ):
        """
        Initialize robot visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            figsize: Figure size
            dpi: DPI for saved figures
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.state_history = []
        self.action_history = []
    
    def log_state(self, state: Dict[str, Any]):
        """
        Log robot state.
        
        Args:
            state: Dictionary containing robot state
        """
        self.state_history.append(state)
    
    def log_action(self, action: np.ndarray):
        """
        Log robot action.
        
        Args:
            action: Action vector
        """
        self.action_history.append(action.copy())
    
    def plot_joint_positions(
        self,
        joint_names: Optional[List[str]] = None,
        show: bool = False,
        save: bool = True,
        filename: str = "joint_positions.png"
    ):
        """
        Plot joint positions over time.
        
        Args:
            joint_names: List of joint names (optional)
            show: Whether to show the plot
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.state_history:
            print("No state data to plot")
            return
        
        joint_positions = []
        for state in self.state_history:
            if 'joint_positions' in state:
                joint_positions.append(state['joint_positions'])
        
        if not joint_positions:
            print("No joint position data")
            return
        
        joint_positions = np.array(joint_positions)
        n_joints = joint_positions.shape[1]
        
        if joint_names is None:
            joint_names = [f'Joint {i}' for i in range(n_joints)]
        
        fig, axes = plt.subplots(n_joints, 1, figsize=(self.figsize[0], self.figsize[1] * n_joints / 4), dpi=self.dpi)
        
        if n_joints == 1:
            axes = [axes]
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_joints))
        
        for i, (ax, name) in enumerate(zip(axes, joint_names)):
            ax.plot(joint_positions[:, i], color=colors[i], linewidth=1.5)
            ax.set_ylabel(name, fontsize=10)
            ax.set_xlabel('Time Step', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved joint positions plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_trajectory(
        self,
        show: bool = False,
        save: bool = True,
        filename: str = "trajectory.png"
    ):
        """
        Plot robot trajectory (x-y position).
        
        Args:
            show: Whether to show the plot
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.state_history:
            print("No state data to plot")
            return
        
        x_positions = []
        y_positions = []
        z_positions = []
        
        for state in self.state_history:
            if 'x_position' in state:
                x_positions.append(state['x_position'])
            if 'y_position' in state:
                y_positions.append(state['y_position'])
            if 'z_position' in state:
                z_positions.append(state['z_position'])
        
        if not x_positions:
            print("No position data")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        ax1 = axes[0]
        ax1.plot(x_positions, y_positions, 'b-', linewidth=1.5, alpha=0.7)
        ax1.scatter(x_positions[0], y_positions[0], c='green', s=100, label='Start', zorder=5)
        ax1.scatter(x_positions[-1], y_positions[-1], c='red', s=100, label='End', zorder=5)
        ax1.set_xlabel('X Position', fontsize=12)
        ax1.set_ylabel('Y Position', fontsize=12)
        ax1.set_title('Robot Trajectory (X-Y Plane)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        ax2 = axes[1]
        time_steps = np.arange(len(z_positions))
        ax2.plot(time_steps, z_positions, 'g-', linewidth=1.5)
        ax2.axhline(y=np.mean(z_positions), color='r', linestyle='--', label=f'Mean: {np.mean(z_positions):.2f}')
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Z Position (Height)', fontsize=12)
        ax2.set_title('Robot Height Over Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved trajectory plot to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_velocity_profile(
        self,
        show: bool = False,
        save: bool = True,
        filename: str = "velocity_profile.png"
    ):
        """
        Plot velocity profile over time.
        
        Args:
            show: Whether to show the plot
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.state_history:
            print("No state data to plot")
            return
        
        x_velocities = []
        z_velocities = []
        
        for state in self.state_history:
            if 'x_velocity' in state:
                x_velocities.append(state['x_velocity'])
            if 'base_linear_velocity' in state:
                vel = state['base_linear_velocity']
                if len(vel) >= 3:
                    z_velocities.append(vel[2])
        
        if not x_velocities:
            print("No velocity data")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        time_steps = np.arange(len(x_velocities))
        ax.plot(time_steps, x_velocities, 'b-', linewidth=1.5, label='X Velocity')
        
        if z_velocities:
            ax.plot(time_steps[:len(z_velocities)], z_velocities, 'g-', linewidth=1.5, label='Z Velocity')
        
        ax.axhline(y=np.mean(x_velocities), color='r', linestyle='--', label=f'Mean X Vel: {np.mean(x_velocities):.2f}')
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Velocity', fontsize=12)
        ax.set_title('Velocity Profile', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved velocity profile to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_gait_analysis(
        self,
        show: bool = False,
        save: bool = True,
        filename: str = "gait_analysis.png"
    ):
        """
        Plot gait analysis metrics.
        
        Args:
            show: Whether to show the plot
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.state_history:
            print("No state data to plot")
            return
        
        z_positions = []
        x_velocities = []
        is_healthy = []
        
        for state in self.state_history:
            if 'z_position' in state:
                z_positions.append(state['z_position'])
            if 'x_velocity' in state:
                x_velocities.append(state['x_velocity'])
            if 'is_healthy' in state:
                is_healthy.append(state['is_healthy'])
        
        if not z_positions:
            print("No gait data")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        time_steps = np.arange(len(z_positions))
        
        ax1 = axes[0, 0]
        ax1.plot(time_steps, z_positions, 'b-', linewidth=1.5)
        ax1.set_xlabel('Time Step', fontsize=10)
        ax1.set_ylabel('Height (Z)', fontsize=10)
        ax1.set_title('Vertical Oscillation', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        if x_velocities:
            ax2.plot(time_steps[:len(x_velocities)], x_velocities, 'g-', linewidth=1.5)
        ax2.set_xlabel('Time Step', fontsize=10)
        ax2.set_ylabel('Forward Velocity', fontsize=10)
        ax2.set_title('Forward Velocity', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        if len(z_positions) > 1:
            z_velocity = np.diff(z_positions)
            ax3.plot(time_steps[:-1], z_velocity, 'r-', linewidth=1.5)
        ax3.set_xlabel('Time Step', fontsize=10)
        ax3.set_ylabel('Vertical Velocity', fontsize=10)
        ax3.set_title('Vertical Velocity', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        if is_healthy:
            healthy_ratio = np.mean(is_healthy)
            ax4.bar(['Healthy', 'Unhealthy'], [healthy_ratio, 1 - healthy_ratio], color=['green', 'red'], alpha=0.7)
            ax4.set_ylabel('Ratio', fontsize=10)
            ax4.set_title('Health Status', fontsize=12, fontweight='bold')
            ax4.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved gait analysis to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_animation(
        self,
        interval: int = 50,
        save: bool = True,
        filename: str = "robot_animation.mp4"
    ):
        """
        Create animation of robot movement.
        
        Args:
            interval: Interval between frames in ms
            save: Whether to save the animation
            filename: Filename for saved animation
        """
        if not self.state_history:
            print("No state data to animate")
            return
        
        x_positions = []
        y_positions = []
        z_positions = []
        
        for state in self.state_history:
            if 'x_position' in state:
                x_positions.append(state['x_position'])
            if 'y_position' in state:
                y_positions.append(state['y_position'])
            if 'z_position' in state:
                z_positions.append(state['z_position'])
        
        if not x_positions:
            print("No position data for animation")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        ax.set_xlim(min(x_positions) - 1, max(x_positions) + 1)
        ax.set_ylim(-1, max(z_positions) + 1)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Z Position (Height)', fontsize=12)
        ax.set_title('Humanoid Robot Animation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=2)
        
        robot_body, = ax.plot([], [], 'ro', markersize=20, alpha=0.8)
        trajectory_line, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5)
        
        def init():
            robot_body.set_data([], [])
            trajectory_line.set_data([], [])
            return robot_body, trajectory_line
        
        def animate(i):
            robot_body.set_data([x_positions[i]], [z_positions[i]])
            trajectory_line.set_data(x_positions[:i+1], z_positions[:i+1])
            return robot_body, trajectory_line
        
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(x_positions), interval=interval, blit=True
        )
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=30, metadata=dict(artist='Humanoid RL'), bitrate=1800)
            anim.save(filepath, writer=writer)
            print(f"Saved animation to {filepath}")
        
        plt.close()
    
    def create_robot_dashboard(
        self,
        show: bool = False,
        save: bool = True,
        filename: str = "robot_dashboard.png"
    ):
        """
        Create a comprehensive robot dashboard.
        
        Args:
            show: Whether to show the plot
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.state_history:
            print("No state data to visualize")
            return
        
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, :])
        
        x_positions = []
        y_positions = []
        z_positions = []
        x_velocities = []
        
        for state in self.state_history:
            if 'x_position' in state:
                x_positions.append(state['x_position'])
            if 'y_position' in state:
                y_positions.append(state['y_position'])
            if 'z_position' in state:
                z_positions.append(state['z_position'])
            if 'x_velocity' in state:
                x_velocities.append(state['x_velocity'])
        
        if x_positions and y_positions:
            ax1.plot(x_positions, y_positions, 'b-', linewidth=1.5, alpha=0.7)
            ax1.scatter(x_positions[0], y_positions[0], c='green', s=100, label='Start')
            ax1.scatter(x_positions[-1], y_positions[-1], c='red', s=100, label='End')
            ax1.set_xlabel('X Position', fontsize=10)
            ax1.set_ylabel('Y Position', fontsize=10)
            ax1.set_title('Robot Trajectory', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
        
        if z_positions:
            time_steps = np.arange(len(z_positions))
            ax2.plot(time_steps, z_positions, 'g-', linewidth=1.5)
            ax2.axhline(y=np.mean(z_positions), color='r', linestyle='--')
            ax2.set_xlabel('Time Step', fontsize=10)
            ax2.set_ylabel('Height (Z)', fontsize=10)
            ax2.set_title('Robot Height', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        if x_velocities:
            time_steps = np.arange(len(x_velocities))
            ax3.plot(time_steps, x_velocities, 'b-', linewidth=1.5)
            ax3.axhline(y=np.mean(x_velocities), color='r', linestyle='--')
            ax3.set_xlabel('Time Step', fontsize=10)
            ax3.set_ylabel('Forward Velocity', fontsize=10)
            ax3.set_title('Forward Velocity', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        if self.action_history:
            actions = np.array(self.action_history)
            if actions.ndim > 1:
                for i in range(min(actions.shape[1], 5)):
                    ax4.plot(actions[:, i], label=f'Action {i}', alpha=0.7)
                ax4.set_xlabel('Time Step', fontsize=10)
                ax4.set_ylabel('Action Value', fontsize=10)
                ax4.set_title('Action Values (First 5)', fontsize=12, fontweight='bold')
                ax4.legend(fontsize=9)
                ax4.grid(True, alpha=0.3)
        
        fig.suptitle('Robot State Dashboard', fontsize=16, fontweight='bold')
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved robot dashboard to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def reset(self):
        """Reset the visualizer state."""
        self.state_history = []
        self.action_history = []
