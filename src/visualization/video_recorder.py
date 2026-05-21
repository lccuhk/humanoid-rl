"""Video Recorder for Humanoid RL"""

import os
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime


class VideoRecorder:
    """
    Video recorder for humanoid robot simulation.
    
    This class provides functionality to record videos of the robot
    during training and evaluation.
    """
    
    def __init__(
        self,
        output_dir: str = "./logs/videos",
        fps: int = 30,
        width: int = 640,
        height: int = 480,
    ):
        """
        Initialize video recorder.
        
        Args:
            output_dir: Directory to save videos
            fps: Frames per second
            width: Video width
            height: Video height
        """
        self.output_dir = output_dir
        self.fps = fps
        self.width = width
        self.height = height
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.frames: List[np.ndarray] = []
        self.is_recording = False
    
    def start_recording(self):
        """Start recording a new video."""
        self.frames = []
        self.is_recording = True
    
    def record_frame(self, frame: np.ndarray):
        """
        Record a single frame.
        
        Args:
            frame: RGB frame as numpy array (H, W, 3)
        """
        if self.is_recording:
            self.frames.append(frame.copy())
    
    def stop_recording(
        self,
        filename: Optional[str] = None,
        format: str = "mp4"
    ) -> str:
        """
        Stop recording and save the video.
        
        Args:
            filename: Output filename (without extension)
            format: Video format ('mp4', 'gif', 'avi')
            
        Returns:
            Path to saved video
        """
        if not self.frames:
            print("Warning: No frames recorded")
            return ""
        
        self.is_recording = False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_{timestamp}"
        
        filepath = os.path.join(self.output_dir, f"{filename}.{format}")
        
        try:
            import cv2
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                filepath,
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            
            for frame in self.frames:
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"Video saved to: {filepath}")
            
        except ImportError:
            print("OpenCV not available, saving as GIF instead...")
            filepath = self._save_as_gif(filename)
        
        self.frames = []
        return filepath
    
    def _save_as_gif(self, filename: str) -> str:
        """Save frames as GIF (fallback method)."""
        try:
            from PIL import Image
            
            filepath = os.path.join(self.output_dir, f"{filename}.gif")
            
            images = []
            for frame in self.frames:
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                images.append(Image.fromarray(frame))
            
            if images:
                images[0].save(
                    filepath,
                    save_all=True,
                    append_images=images[1:],
                    duration=1000 // self.fps,
                    loop=0
                )
                print(f"GIF saved to: {filepath}")
            
            return filepath
            
        except ImportError:
            print("PIL not available, cannot save video")
            return ""
    
    def record_episode(
        self,
        env,
        agent,
        n_steps: int = 1000,
        deterministic: bool = True,
        filename: Optional[str] = None,
    ) -> str:
        """
        Record an episode of agent interaction with environment.
        
        Args:
            env: Gymnasium environment
            agent: RL agent with predict method
            n_steps: Maximum number of steps to record
            deterministic: Whether to use deterministic actions
            filename: Output filename
            
        Returns:
            Path to saved video
        """
        self.start_recording()
        
        obs, _ = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < n_steps:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if hasattr(env, 'render') and env.render_mode == 'rgb_array':
                frame = env.render()
                if frame is not None:
                    self.record_frame(frame)
            
            done = terminated or truncated
            step_count += 1
        
        return self.stop_recording(filename)
    
    def get_frame_count(self) -> int:
        """Get number of recorded frames."""
        return len(self.frames)
    
    def clear_frames(self):
        """Clear all recorded frames."""
        self.frames = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recorder statistics."""
        return {
            'output_dir': self.output_dir,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'current_frames': len(self.frames),
            'is_recording': self.is_recording,
        }
