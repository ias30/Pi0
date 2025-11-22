import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_realman_example() -> dict:
    """Creates a random input example for the Realman policy."""
    return {
        "state": np.ones((12,)),  # 12-DoF: right arm (6) + left arm (6)
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "Let the forceps go along the black S shaped path",
    }


@dataclasses.dataclass(frozen=True)
class RealmanInputs(transforms.DataTransformFn):
    """Inputs for the Realman policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [12] - right arm (6 joints) + left arm (6 joints), no grippers
    - actions: [action_horizon, 12] - right arm (6 joints) + left arm (6 joints), no grippers
    """

    # If true, this will convert the joint and gripper values from the standard Realman space to
    # the space used by the pi internal runtime which was used to train the base model.
    # Set to False if your Realman data is already in the correct format.
    adapt_to_pi: bool = False

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_realman(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if self.adapt_to_pi:
                actions = _encode_actions_inv(actions)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RealmanOutputs(transforms.DataTransformFn):
    """Outputs for the Realman policy."""

    # If true, this will convert the joint values from the pi space back to
    # the standard Realman space.
    adapt_to_pi: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first 12 dims (right arm 6 + left arm 6).
        actions = np.asarray(data["actions"][:, :12])
        if self.adapt_to_pi:
            actions = _encode_actions(actions)
        return {"actions": actions}


def _decode_realman(data: dict, *, adapt_to_pi: bool = False) -> dict:
    """Decode Realman data format.
    
    Args:
        data: Input data containing state and images
        adapt_to_pi: If True, applies transformations to match pi0 training format
    
    Returns:
        Decoded data with properly formatted state and images
    """
    # state is [right_arm_joint_angles, left_arm_joint_angles]
    # dim sizes: [6 joints, 6 joints] = 12 total, no grippers
    # Right arm joints: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
    # Left arm joints: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
    state = np.asarray(data["state"])
    if adapt_to_pi:
        state = _decode_state(state)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray) -> np.ndarray:
    """Apply transformations to state if needed.
    
    Modify this function if you need to apply specific transformations
    to your Realman robot state to match the pi0 training format.
    """
    # Add any Realman-specific state transformations here if needed
    return state


def _encode_actions(actions: np.ndarray) -> np.ndarray:
    """Encode actions from pi0 format to Realman format.
    
    Modify this function if you need to apply specific transformations
    to convert from pi0 action space to your Realman robot action space.
    """
    # Add any Realman-specific action transformations here if needed
    return actions


def _encode_actions_inv(actions: np.ndarray) -> np.ndarray:
    """Encode actions from Realman format to pi0 format (inverse of _encode_actions).
    
    Modify this function if you need to apply specific transformations
    to convert from Realman action space to pi0 training format.
    """
    # Add any Realman-specific action transformations here if needed
    return actions

