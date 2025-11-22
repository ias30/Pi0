# Pi0
#Step 1:convert realman_data to lerobotv2.1
```json
{
    "codebase_version": "v2.0",
    "robot_type": "aloha",
    "total_episodes": 123,
    "total_frames": 36900,
    "total_tasks": 1,
    "total_videos": 0,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": 50,
    "splits": {
        "train": "0:123"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "observation.state": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": [
                [
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper",
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper"
                ]
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": [
                [
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper",
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper"
                ]
            ]
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": [
                [
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper",
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper"
                ]
            ]
        },
        "observation.effort": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": [
                [
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper",
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper"
                ]
            ]
        },
        "observation.images.cam_high": {
            "dtype": "image",
            "shape": [
                3,
                480,
                640
            ],
            "names": [
                "channels",
                "height",
                "width"
            ]
        },
        "observation.images.cam_low": {
            "dtype": "image",
            "shape": [
                3,
                480,
                640
            ],
            "names": [
                "channels",
                "height",
                "width"
            ]
        },
        "observation.images.cam_left_wrist": {
            "dtype": "image",
            "shape": [
                3,
                480,
                640
            ],
            "names": [
                "channels",
                "height",
                "width"
            ]
        },
        "observation.images.cam_right_wrist": {
            "dtype": "image",
            "shape": [
                3,
                480,
                640
            ],
            "names": [
                "channels",
                "height",
                "width"
            ]
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    }
}```

#Step 2:Defining training configs and running training
a. LiberoInputs and LiberoOutputs: Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
Defines the data mappings between the LIBERO environment and the model, in both directions.These mappings are used for both training and inference.


b. LeRobotLiberoDataConfig: Defines how to process raw LIBERO data from LeRobot dataset for training.
A configuration dedicated to “training data,” specifying whether multiple camera views should be combined, and how the raw LIBERO data from a LeRobot dataset should be processed for training.

c. TrainConfig: Defines fine-tuning hyperparameters, data config, and weight loader.
TrainConfig serves as the master controller, defining training hyperparameters such as learning rate, batch size, and related settings.


#Step 3:Spinning up a policy server and running inference(offline/realtime inference)
action_chunk=25

