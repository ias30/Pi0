#!/usr/bin/env python3
"""Real-time inference script for Realman π₀ model.

This script performs live inference with dual-arm Realman robots, using camera inputs
and joint states to predict actions. It executes actions in a receding horizon manner
(predict H steps, execute k steps, then replan).

Usage:
    uv run /home/ren9/DualArm/openpi/examples/realman_inference.py \
        --config-name pi0_realman \
        --checkpoint checkpoints/pi0_realman/realman_finetune_v1/14999/params \
        --norm-stats assets/pi0_realman/realman_dataset/ \
        --output inference_actions.csv \
        --speed 20 \
        --steps-to-execute 3
"""

import argparse
import csv
import logging
import pathlib
import signal
import sys
import time
from typing import Any

import cv2
import jax
print("JAX default backend:", jax.default_backend())
print("JAX devices:", jax.devices())
import jax.numpy as jnp
import numpy as np
from PIL import Image

# Add data collection module to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data_collection_pi0"))

from Robotic_Arm.rm_robot_interface import *
from camera_collector import TripleCameraCollector

import openpi.models.model as _model
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.transforms as _transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

# def resize_with_pad(images: np.ndarray, height: int, width: int) -> np.ndarray:
#     """Resize images with padding to maintain aspect ratio.
    
#     Args:
#         images: Input images in HWC format, shape (H, W, C)
#         height: Target height
#         width: Target width
        
#     Returns:
#         Resized image with padding, shape (height, width, C)
#     """
#     #read time first
#     resize_start_time = time.time()
#     if images.shape[0] == height and images.shape[1] == width:
#         return images
    
#     pil_image = Image.fromarray(images)
#     cur_width, cur_height = pil_image.size
    
#     ratio = max(cur_width / width, cur_height / height)
#     resized_height = int(cur_height / ratio)
#     resized_width = int(cur_width / ratio)
#     resized_image = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    
#     # Create black canvas and paste resized image
#     zero_image = Image.new(resized_image.mode, (width, height), 0)
#     pad_height = max(0, int((height - resized_height) / 2))
#     pad_width = max(0, int((width - resized_width) / 2))
#     zero_image.paste(resized_image, (pad_width, pad_height))
#     resize_time= time.time() - resize_start_time
#     print(f"Resize time: {resize_time:.4f} seconds")
#     return np.array(zero_image)


def preprocess_image(image_bgr: np.ndarray, target_size=None) -> np.ndarray:
    """Preprocess camera image for model input.
    
    Applies BGR->RGB conversion, CHW transpose, and normalization to [0, 1] to match
    LeRobot dataset format (mode="image").
    
    Args:
        image_bgr: BGR image from camera, shape (H, W, 3), uint8 [0, 255]
        target_size: Target image size (height, width) - not used, kept for compatibility
        
    Returns:
        RGB image in CHW format, shape (3, H, W), float32 [0, 1]
    """
    preprocess_start_time = time.time()
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to CHW format - required by RealmanInputs._decode_realman
    # which expects CHW and will convert to HWC internally
    image_chw = np.transpose(image_rgb, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    
    # Normalize to [0, 1] to match LeRobot dataset format (mode="image")
    # This matches how LeRobot processes images when dtype="image"
    # RealmanInputs._decode_realman will detect float dtype and convert back to uint8 by * 255
    image_chw = image_chw.astype(np.float32) / 255.0  # [0, 255] -> [0, 1]
    
    preprocess_time = time.time() - preprocess_start_time
    print(f"Preprocess time: {preprocess_time:.4f} seconds")
    print(f"  Output shape: {image_chw.shape}, dtype: {image_chw.dtype}")
    return image_chw


def deg_to_rad(angles_deg: np.ndarray) -> np.ndarray:
    """Convert angles from degrees to radians."""
    return np.deg2rad(angles_deg)


def rad_to_deg(angles_rad: np.ndarray) -> np.ndarray:
    """Convert angles from radians to degrees."""
    return np.rad2deg(angles_rad)


# ============================================================================
# Main Inference Class
# ============================================================================

class RealmanInference:
    """Real-time inference system for Realman dual-arm robots."""
    
    def __init__(
        self,
        config_name: str,
        checkpoint_path: str,
        norm_stats_path: str,
        output_csv_path: str,
        speed: int = 20,
        steps_to_execute: int = 3,
        enable_visualization: bool = False,
    ):
        """Initialize the inference system.
        
        Args:
            config_name: Name of the training config
            checkpoint_path: Path to model checkpoint
            norm_stats_path: Path to normalization statistics
            output_csv_path: Path to save executed actions
            speed: Robot movement speed (1-100)
            steps_to_execute: Number of action steps to execute before replanning
            enable_visualization: Whether to show camera feeds (requires GUI support)
        """
        self.speed = speed
        self.steps_to_execute = steps_to_execute
        self.output_csv_path = pathlib.Path(output_csv_path)
        self.running = True
        self.prompt = "Let the forceps go along the black S shaped path"
        self.enable_visualization = enable_visualization
        print("1"*80)
        # RNG for model sampling
        self.rng = jax.random.key(42)
        
        # Robot IP addresses
        self.left_robot_ip = "169.254.128.18"
        self.right_robot_ip = "169.254.128.19"
        
        # Initial positions (in degrees)
        self.initial_right_angles = [80.212, 40.867, 66.459, -45.413, -17.694, 0.480]
        self.initial_left_angles = [-77.58322164,-43.524,-70.19155,79.19366304,15.6338225,33.85788]
        # self.initial_left_angles = [-79.5739974975586, -44.62900161743164, -71.25800323486328, 
        #                             70.08100128173828, 14.241999626159668, 42.44200134277344]
        
        print("=" * 80)
        print("🚀 Initializing Realman π₀ Real-time Inference System")
        print("=" * 80)
        
        # Setup signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Load model and config
        self.model, self.train_config, self.data_config = self._load_model(
            config_name, checkpoint_path
        )
        
        # Load normalization stats
        self.norm_stats = self._load_norm_stats(norm_stats_path)
        
        # Create transforms
        self.input_transform, self.output_transform = self._create_transforms()
        
        # Initialize hardware
        self._initialize_hardware()
        
        # Prepare CSV file
        self._prepare_csv_file()
        
        print("✅ Initialization complete!")
        print("=" * 80)
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n🛑 Ctrl+C detected. Stopping inference...")
        self.running = False
    
    def _load_model(self, config_name: str, checkpoint_path: str):
        """Load model and configuration."""
        print(f"Loading config: {config_name}")
        train_config = _config.get_config(config_name)
        
        # Create data config
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        
        # Load checkpoint parameters
        print(f"Loading checkpoint from: {checkpoint_path}")
        loaded_params = _model.restore_params(checkpoint_path, restore_type=np.ndarray)
        
        # Create model with loaded parameters
        print(f"Creating model: {train_config.model.model_type}")
        model = train_config.model.load(loaded_params)
        print("2"*80)
        return model, train_config, data_config
    
    def _load_norm_stats(self, norm_stats_path: str):
        """Load normalization statistics."""
        path = pathlib.Path(norm_stats_path)
        if not path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {path}")
        
        print(f"Loading normalization stats: {path}")
        print("3"*80)
        return _normalize.load(path)
    
    def _create_transforms(self):
        """Create input and output transforms."""
        # Input transforms: repack → data → normalize → model
        input_transforms = _transforms.compose([
            *self.data_config.repack_transforms.inputs,
            *self.data_config.data_transforms.inputs,
            _transforms.Normalize(self.norm_stats, use_quantiles=self.data_config.use_quantile_norm),
            *self.data_config.model_transforms.inputs,
        ])
        
        # Output transforms: model → unnormalize → data
        output_transforms = _transforms.compose([
            *self.data_config.model_transforms.outputs,
            _transforms.Unnormalize(self.norm_stats, use_quantiles=self.data_config.use_quantile_norm),
            *self.data_config.data_transforms.outputs,
        ])
        print("4"*80)
        return input_transforms, output_transforms
    
    def _initialize_hardware(self):
        """Initialize robot arms and cameras."""
        print("5"*80)
        print("🤖 Initializing robot arms...")
        
        # Initialize left arm (169.254.128.18)
        self.left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        left_result = self.left_arm.rm_create_robot_arm(self.left_robot_ip, 8080)
        if left_result.id != 1:
            raise RuntimeError(f"Failed to connect to left arm: {left_result}")
        print(f"✅ Left arm connected: {self.left_robot_ip}")
        print("6"*80)
        
        # Initialize right arm (169.254.128.19)
        self.right_arm = RoboticArm()
        right_result = self.right_arm.rm_create_robot_arm(self.right_robot_ip, 8080)
        if right_result.id != 2:
            raise RuntimeError(f"Failed to connect to right arm: {right_result}")
        print(f"✅ Right arm connected: {self.right_robot_ip}")
        print("7"*80)
        
        # Move to initial positions
        print("🦾 Moving to initial positions...")
        print("  Moving right arm...")
        self.right_arm.rm_movej(self.initial_right_angles, self.speed, 0, 0, True)
        print("  ✅ Right arm at initial position")
        
        print("  Moving left arm...")
        self.left_arm.rm_movej(self.initial_left_angles, self.speed, 0, 0, True)
        print("  ✅ Left arm at initial position")
        
        time.sleep(0.5)  # Wait for stabilization
        
        # Initialize cameras
        print("📷 Initializing cameras...")
        self.camera_collector = TripleCameraCollector(resolution=(640, 480), fps=30)
        
        if not self.camera_collector.connect_all():
            logger.warning("⚠️  Some cameras failed to connect, but continuing...")
        
        if not self.camera_collector.start_all():
            logger.warning("⚠️  Some cameras failed to start, but continuing...")
        
        time.sleep(1)  # Allow cameras to warm up
        print("✅ Cameras initialized")
    
    def _prepare_csv_file(self):
        """Prepare CSV file for saving executed actions."""
        # Create header
        print("prepare csv file")
        joint_names = [
            "joint_0_right_waist", "joint_1_right_shoulder", "joint_2_right_elbow",
            "joint_3_right_forearm_roll", "joint_4_right_wrist_angle", "joint_5_right_wrist_rotate",
            "joint_6_left_waist", "joint_7_left_shoulder", "joint_8_left_elbow",
            "joint_9_left_forearm_roll", "joint_10_left_wrist_angle", "joint_11_left_wrist_rotate",
        ]
        
        self.csv_header = joint_names + ["timestamp"]
        
        # Create file and write header
        with open(self.output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)
        
        print(f"📝 CSV file prepared: {self.output_csv_path}")
    
    def _get_current_state(self) -> dict:
        """Get current robot state and camera images.
        
        Returns:
            Dictionary with state (joint angles) and images
        """
        arm_start_time = time.time()
        # Get joint angles from both arms
        left_result, left_state = self.left_arm.rm_get_current_arm_state()
        right_result, right_state = self.right_arm.rm_get_current_arm_state()
        print("left_state:", left_state)
        if left_result != 0 or right_result != 0:
            raise RuntimeError("Failed to get arm states")
        
        # Extract joint angles (in degrees)
        left_joints_deg = np.array(left_state['joint'][:6])
        right_joints_deg = np.array(right_state['joint'][:6])
        
        # Convert to radians and concatenate (right + left)
        left_joints_rad = deg_to_rad(left_joints_deg)
        right_joints_rad = deg_to_rad(right_joints_deg)
        state_rad = np.concatenate([right_joints_rad, left_joints_rad])  # (12,)
        print("state_rad:", state_rad)
        arm_time = time.time() - arm_start_time
        print(f"Arm state retrieval time: {arm_time:.4f} seconds")
        # Get camera images
        camera_start_time = time.time()
        _, cam_high_img, _ = self.camera_collector.camera_high.get_latest_data()
        _, cam_left_wrist_img = self.camera_collector.camera_left_wrist.get_latest_data()
        _, cam_right_wrist_img = self.camera_collector.camera_right_wrist.get_latest_data()
        camera_time = time.time() - camera_start_time
        print(f"Camera image retrieval time: {camera_time:.4f} seconds")
        
        # Preprocess images
        preprocess_start_time = time.time()
        cam_high_processed = preprocess_image(cam_high_img)
        cam_left_wrist_processed = preprocess_image(cam_left_wrist_img)
        cam_right_wrist_processed = preprocess_image(cam_right_wrist_img)
        preprocess_time = time.time() - preprocess_start_time
        print(f"Preprocess time: {preprocess_time:.4f} seconds")
        
        # Create placeholder action for inference (required by transforms but not used)
        # Shape: (action_horizon, 12) - will be filled by the model prediction
        placeholder_action = np.zeros((self.train_config.model.action_horizon, 12), dtype=np.float32)

        # Return in LeRobot dataset format
        state_dict = {
            "observation.state": state_rad,
            "action": placeholder_action,  # Placeholder for transforms
            "observation.images.cam_high": cam_high_processed,
            "observation.images.cam_left_wrist": cam_left_wrist_processed,
            "observation.images.cam_right_wrist": cam_right_wrist_processed,
            "prompt": self.prompt,
            "raw_images": {
                "cam_high": cam_high_img,
                "cam_left_wrist": cam_left_wrist_img,
                "cam_right_wrist": cam_right_wrist_img,
            }
        }
        
        # 1. DEBUG: Print state_dict after data acquisition
        print("\n=== 1. After Data Acquisition ===")
        # Check image value range
        for img_key in ['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']:
            if img_key in state_dict:
                img = state_dict[img_key]
                if isinstance(img, np.ndarray):
                    print(f"  [IMAGE VALUES] {img_key}: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}")
        print("=" * 50)
        for key, value in state_dict.items():
            if key != "raw_images":
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: type={type(value)}, value={value if not isinstance(value, str) or len(value) < 50 else value[:50]+'...'}")
        print("=" * 50)
        
        return state_dict
    
    def _run_inference(self, state_dict: dict) -> np.ndarray:
        """Run model inference on current state.
        
        Args:
            state_dict: Dictionary with state and images
            
        Returns:
            Predicted actions in radians, shape (action_horizon, 12)
        """
        print("Running inference...")
        
        # 2. DEBUG: Print state_dict before input transform
        print("\n=== 2. Before Input Transform ===")
        for key, value in state_dict.items():
            if key != "raw_images":
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: type={type(value)}, value={value if not isinstance(value, str) or len(value) < 50 else value[:50]+'...'}")
        print("=" * 50)
        
        # Transform input
        inference_start_time = time.time()
        transformed = self.input_transform(state_dict)
        
        # 3. DEBUG: Print transformed data after input transform
        print("\n=== 3. After Input Transform ===")
        for key, value in transformed.items():
            if isinstance(value, dict):
                print(f"  {key}: (dict with {len(value)} keys)")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        print(f"    {sub_key}: shape={sub_value.shape}, dtype={sub_value.dtype}")
                    else:
                        print(f"    {sub_key}: type={type(sub_value)}")
            elif isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: type={type(value)}")
        print("=" * 50)
        
        # Add batch dimension (use numpy arrays)
        batch = jax.tree.map(lambda x: np.expand_dims(x, axis=0), transformed)
        
        # Convert numpy arrays to JAX arrays (required by the model)
        batch = jax.tree.map(lambda x: jnp.asarray(x), batch)
        
        # Create observation
        observation = _model.Observation.from_dict(batch)
        
        # 4. DEBUG: Print observation after creation
        print("\n=== 4. After Observation Creation ===")
        print(f"  images: (dict with {len(observation.images)} keys)")
        for key, value in observation.images.items():
            arr = np.array(value)
            print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"  image_masks: (dict with {len(observation.image_masks)} keys)")
        for key, value in observation.image_masks.items():
            arr = np.array(value)
            print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")
        state_arr = np.array(observation.state)
        print(f"  state: shape={state_arr.shape}, dtype={state_arr.dtype}")
        if observation.tokenized_prompt is not None:
            token_arr = np.array(observation.tokenized_prompt)
            print(f"  tokenized_prompt: shape={token_arr.shape}, dtype={token_arr.dtype}")
        if observation.tokenized_prompt_mask is not None:
            mask_arr = np.array(observation.tokenized_prompt_mask)
            print(f"  tokenized_prompt_mask: shape={mask_arr.shape}, dtype={mask_arr.dtype}")
        print("=" * 50)
        
        # Run model inference using sample_actions
        # Split RNG for this inference step
        self.rng, inference_rng = jax.random.split(self.rng)
        predicted_actions = self.model.sample_actions(
            inference_rng,
            observation,
            num_steps=10  # Number of diffusion steps for Pi0
        )  # Shape: (1, action_horizon, 12)
        
        # Remove batch dimension
        predicted_actions = predicted_actions[0]  # (action_horizon, 12)
        
        # Convert delta actions to absolute actions and unnormalize
        # CRITICAL: Must pass both state and actions for delta-to-absolute conversion
        # Add action_horizon dimension for each action step
        state_np = np.array(transformed["state"])  # Get the state
        
        # Process each action in the horizon
        absolute_actions = []
        for step_idx in range(predicted_actions.shape[0]):
            # Prepare data for transform (needs 2D actions: [1, action_dim])
            denorm_dict = {
                "actions": np.array(predicted_actions[step_idx])[np.newaxis, :],  # Shape: (1, 12)
                "state": state_np  # Pass current state for delta-to-absolute conversion
            }
            
            # Apply output transforms (unnormalize + delta->absolute)
            denorm_result = self.output_transform(denorm_dict)
            
            # Extract the action and remove extra dimension
            absolute_actions.append(denorm_result["actions"][0])  # Shape: (12,)
        
        inference_time = time.time() - inference_start_time
        print(f"Inference time: {inference_time:.4f} seconds")
        
        return np.array(absolute_actions)  # Shape: (action_horizon, 12)
    
    def _execute_action(self, action_rad: np.ndarray):
        """Execute a single action on both arms.
        
        Args:
            action_rad: Action in radians, shape (12,)
        """
        # Split into right and left
        right_action_rad = action_rad[:6]
        left_action_rad = action_rad[6:]
        
        # Convert to degrees
        right_action_deg = rad_to_deg(right_action_rad).tolist()
        left_action_deg = rad_to_deg(left_action_rad).tolist()
        
        # Execute on both arms (blocking)
        self.right_arm.rm_movej(right_action_deg, self.speed, 0, 0, True)
        self.left_arm.rm_movej(left_action_deg, self.speed, 0, 0, True)
        # time.sleep(1.0)  # Small delay to ensure commands are sent
    
    def _save_action_to_csv(self, action_rad: np.ndarray, timestamp: float):
        """Save executed action to CSV file.
        
        Args:
            action_rad: Action in radians, shape (12,)
            timestamp: Timestamp of execution
        """
        
        # Convert to degrees for CSV
        action_deg = rad_to_deg(action_rad)
        
        # Append to CSV
        with open(self.output_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = list(action_deg) + [timestamp]
            writer.writerow(row)
    
    def _visualize(self, raw_images: dict):
        """Visualize camera feeds.
        
        Args:
            raw_images: Dictionary of raw BGR images
        """
        if not self.enable_visualization:
            return
        
        try:
            # Create combined view
            cam_high = cv2.resize(raw_images["cam_high"], (320, 240))
            cam_left = cv2.resize(raw_images["cam_left_wrist"], (320, 240))
            cam_right = cv2.resize(raw_images["cam_right_wrist"], (320, 240))
            
            # Stack images horizontally
            combined = np.hstack([cam_high, cam_left, cam_right])
            
            # Add labels
            cv2.putText(combined, "High", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "Left Wrist", (330, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "Right Wrist", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Realman π₀ Inference - Camera Feeds", combined)
            cv2.waitKey(1)
        except cv2.error as e:
            # Disable visualization if it fails (no GUI support)
            logger.warning(f"⚠️  Disabling visualization due to error: {e}")
            self.enable_visualization = False
    
    def run(self):
        """Main inference loop."""
        print("=" * 80)
        print("🎮 Starting inference loop")
        print(f"  Action horizon: {self.train_config.model.action_horizon}")
        print(f"  Steps to execute: {self.steps_to_execute}")
        print(f"  Robot speed: {self.speed}")
        print(f"  Output CSV: {self.output_csv_path}")
        print("  Press Ctrl+C to stop")
        print("=" * 80)
        
        iteration = 0
        
        try:
            while self.running:
                iteration += 1
                loop_start_time = time.time()
                
                print(f"\n{'='*60}")
                print(f"Iteration {iteration}")
                print(f"{'='*60}")
                
                # 1. Get current state
                print("📊 Getting current state...")
                state_dict = self._get_current_state()
                
                # 2. Run inference
                print("🧠 Running inference...")
                inference_start = time.time()
                predicted_actions = self._run_inference(state_dict)
                inference_time = time.time() - inference_start
                print(f"  Inference took {inference_time:.3f}s")
                print(f"  Predicted {predicted_actions.shape[0]} steps")
                
                # 3. Execute first k steps
                num_steps = min(self.steps_to_execute, predicted_actions.shape[0])
                print(f"🎯 Executing first {num_steps} steps...")
                
                for step_idx in range(num_steps):
                    action = predicted_actions[step_idx]
                    
                    print(f"  Step {step_idx + 1}/{num_steps}: Executing action...")
                    exec_start = time.time()
                    
                    # Execute action
                    self._execute_action(action)
                    
                    exec_time = time.time() - exec_start
                    timestamp = time.time()
                    
                    # Save to CSV
                    self._save_action_to_csv(action, timestamp)
                    
                    print(f"    ✅ Executed in {exec_time:.3f}s")
                
                # 4. Visualize
                self._visualize(state_dict["raw_images"])
                
                # 5. Stats
                loop_time = time.time() - loop_start_time
                print(f"\n📈 Iteration {iteration} complete:")
                print(f"  Total time: {loop_time:.3f}s")
                print(f"  Inference time: {inference_time:.3f}s")
                print(f"  Execution time: {loop_time - inference_time:.3f}s")
                print(f"  Effective frequency: {1.0 / loop_time:.2f} Hz")
        
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received")
        except Exception as e:
            logger.error(f"\n❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        
        # Stop cameras
        try:
            self.camera_collector.stop_all()
            self.camera_collector.disconnect_all()
            print("  ✅ Cameras stopped")
        except Exception as e:
            logger.error(f"  ❌ Error stopping cameras: {e}")
        
        # Close visualization window
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.warning(f"  ⚠️  Could not destroy OpenCV windows (no GUI support): {e}")
        
        # Note: Robot arms will disconnect automatically when object is destroyed
        print("  ✅ Robot arms cleanup complete")
        
        print("✅ Cleanup complete")
        print(f"📁 Actions saved to: {self.output_csv_path}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Real-time inference for Realman π₀ dual-arm robots"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi0_realman",
        help="Name of the training config (default: pi0_realman)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/ren9/DualArm/openpi/checkpoints/pi0_realman/realman_finetune_v1/14999/params",
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--norm-stats",
        type=str,
        default="/home/ren9/DualArm/openpi/assets/pi0_realman/realman_dataset/",
        help="Path to normalization stats",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_actions.csv",
        help="Output CSV file path for executed actions",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=30,
        help="Robot movement speed (1-100, default: 20)",
    )
    parser.add_argument(
        "--steps-to-execute",
        type=int,
        default=10,
        help="Number of action steps to execute before replanning (default: 20)",
    )
    parser.add_argument(
        "--enable-visualization",
        action="store_true",
        default=False,
        help="Enable camera feed visualization (requires GUI support, default: disabled)",
    )
    
    args = parser.parse_args()
    

    # Create inference system
    inference_system = RealmanInference(
        config_name=args.config_name,
        checkpoint_path=args.checkpoint,
        norm_stats_path=args.norm_stats,
        output_csv_path=args.output,
        speed=args.speed,
        steps_to_execute=args.steps_to_execute,
        enable_visualization=args.enable_visualization,
    )
    
    # Run inference
    inference_system.run()


if __name__ == "__main__":
    main()

