#!/usr/bin/env python3

import roslibpy
import cv2
import numpy as np
import torch
import pickle
import yaml
import os
import time
import base64


# Import ACT code
from policy import ACTPolicy
from utils import TemporalEnsembler


class RoslibpyInference:
    def __init__(self, host="localhost", port=9091):
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Paths
        ckpt_dir = "/home/eai/act/data/checkpoints/fruits_picking"
        config_path = os.path.join(ckpt_dir, "config.yaml")
        ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            policy_config = config["policy_config"]

        # Load checkpoint
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.policy = ACTPolicy(policy_config)
        self.policy.load_state_dict(state_dict)
        self.policy.to(self.device)
        self.policy.eval()

        # Load stats
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

        # Temporal ensembling setup
        self.num_queries = policy_config["num_queries"]
        self.ensembler = TemporalEnsembler(
            chunk_size=self.num_queries, action_dim=7, device=self.device
        )

        # Publishers
        self.action_pub = roslibpy.Topic(
            self.client,
            "/left/leader_arm_control_node/commands",
            "sensor_msgs/msg/JointState",
        )

        # Subscribers
        self.piper_sub = roslibpy.Topic(
            self.client, "/synced/left/piper/joint_states", "sensor_msgs/msg/JointState"
        )
        self.gripper_joint_sub = roslibpy.Topic(
            self.client,
            "/synced/left/gripper_joint_state",
            "sensor_msgs/msg/JointState",
        )
        self.main_img_sub = roslibpy.Topic(
            self.client,
            "/synced/main/image_raw/compressed",
            "sensor_msgs/msg/CompressedImage",
        )
        self.gripper_img_sub = roslibpy.Topic(
            self.client,
            "/synced/gripper/image_raw/compressed",
            "sensor_msgs/msg/CompressedImage",
        )
        self.pace_sub = roslibpy.Topic(
            self.client, "/synced/pace_maker", "sensor_msgs/msg/JointState"
        )

        # Latest messages
        self.latest_piper = None
        self.latest_gripper_joint = None
        self.latest_main_img = None
        self.latest_gripper_img = None

        # Setup callbacks
        self.piper_sub.subscribe(self.piper_callback)
        self.gripper_joint_sub.subscribe(self.gripper_joint_callback)
        self.main_img_sub.subscribe(self.main_img_callback)
        self.gripper_img_sub.subscribe(self.gripper_img_callback)
        self.pace_sub.subscribe(self.pace_callback)

        print("Roslibpy inference initialized")

    def piper_callback(self, msg):
        # print("In piper_callback")
        self.latest_piper = msg

    def gripper_joint_callback(self, msg):
        # print("In gripper_joint_callback")
        self.latest_gripper_joint = msg

    def main_img_callback(self, msg):
        # print("In main_img_callback")
        self.latest_main_img = msg

    def gripper_img_callback(self, msg):
        # print("In gripper_img_callback")
        self.latest_gripper_img = msg

    def pace_callback(self, msg):
        # Check if all messages are recent (within 0.1s)
        # print("In pace_callback")
        if (
            self.latest_piper
            and self.latest_gripper_joint
            and self.latest_main_img
            and self.latest_gripper_img
        ):
            piper_time = (
                self.latest_piper["header"]["stamp"]["sec"]
                + self.latest_piper["header"]["stamp"]["nanosec"] * 1e-9
            )
            gripper_joint_time = (
                self.latest_gripper_joint["header"]["stamp"]["sec"]
                + self.latest_gripper_joint["header"]["stamp"]["nanosec"] * 1e-9
            )
            main_time = (
                self.latest_main_img["header"]["stamp"]["sec"]
                + self.latest_main_img["header"]["stamp"]["nanosec"] * 1e-9
            )
            gripper_time = (
                self.latest_gripper_img["header"]["stamp"]["sec"]
                + self.latest_gripper_img["header"]["stamp"]["nanosec"] * 1e-9
            )
            pace_time = (
                msg["header"]["stamp"]["sec"] + msg["header"]["stamp"]["nanosec"] * 1e-9
            )

            times = [piper_time, gripper_joint_time, main_time, gripper_time, pace_time]
            if max(times) - min(times) <= 1./15.:  # control freq is 15Hz
                self.run_inference(
                    self.latest_piper,
                    self.latest_gripper_joint,
                    self.latest_main_img,
                    self.latest_gripper_img,
                )

    def decompress_image(self, msg):
        """Decompress CompressedImage to RGB numpy array."""
        fmt = msg["format"].lower()
        if "jpeg" in fmt:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(msg["data"]), np.uint8), cv2.IMREAD_COLOR)
            if "bgr8" in fmt:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif "png" in fmt:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(msg["data"]), np.uint8), cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            if "bgr8" in fmt:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image format: {msg['format']}")
        return img

    def process_images(self, image_list):
        """Process list of decompressed images: stack, normalize, return torch tensor."""
        images = []
        for img in image_list:
            images.append(img)
        image = np.stack(images, axis=0)  # (num_imgs, H, W, 3)
        # uint8 to float32, HWC to CHW, /255
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (0, 3, 1, 2))
        return torch.from_numpy(image)

    def run_inference(
        self, piper_msg, gripper_joint_msg, main_img_msg, gripper_img_msg
    ):
        # print("In run_inference")
        # Process joint states
        piper_pos = np.array(piper_msg["position"])  # 6 DoF
        gripper_pos = np.array(gripper_joint_msg["position"])  # 1 DoF
        qpos = np.concatenate([piper_pos, gripper_pos])  # 7 dims
        # Normalize
        qpos = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]

        # Process images
        main_img = self.decompress_image(main_img_msg)
        gripper_img = self.decompress_image(gripper_img_msg)
        image_list = [gripper_img, main_img]  # order as in act_inference
        image = (
            self.process_images(image_list).unsqueeze(0).to(self.device)
        )  # (1, 2, 3, H, W)

        qpos = torch.from_numpy(qpos).float().unsqueeze(0).to(self.device)  # (1, 7)

        # Inference with temporal ensembling
        with torch.inference_mode():
            all_actions = self.policy(qpos, image)  # (1, num_queries, action_dim)
            predicted_chunk = all_actions.squeeze(0)  # (num_queries, 7)
            raw_action = self.ensembler.get_ensembled_action(predicted_chunk)

        # Denormalize action
        action = raw_action.cpu()
        action = action * torch.tensor(self.stats["action_std"]) + torch.tensor(
            self.stats["action_mean"]
        )
        action = action.numpy()

        # Publish
        joint_state = {
            "header": {"stamp": roslibpy.Time.now(), "frame_id": "world"},
            "name": ["joint_" + str(i) for i in range(7)],
            "position": action.tolist(),
            "velocity": [],
            "effort": [],
        }
        self.action_pub.publish(roslibpy.Message(joint_state))

    def run(self):
        try:
            while self.client.is_connected:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.client.terminate()


if __name__ == "__main__":
    inference = RoslibpyInference()
    inference.run()
