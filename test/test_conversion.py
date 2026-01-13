import unittest
from mcap_ros2.reader import read_ros2_messages
import h5py
import numpy as np
import cv2
from pathlib import Path


def decompress_image(msg):
    """Decompress CompressedImage to RGB numpy array."""
    fmt = msg.format.lower()
    if "jpeg" in fmt:
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if "bgr8" in fmt:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    elif "png" in fmt:
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            img = img[:, :, :3]  # Drop alpha
        if "bgr8" in fmt:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        raise ValueError(f"Unsupported image format: {msg.format}")


class TestConversion(unittest.TestCase):

    def setUp(self):
        self.data_dir = Path("/home/eai/data/fruits_picking")
        self.hdf5_dir = Path("/home/eai/act/data/fruits_picking")
        self.indices = [0, 50, 100]

    def test_episode_0(self):
        episode = 0
        folders = sorted([f for f in self.data_dir.iterdir() if f.is_dir()])
        folder = folders[episode]
        mcap_file = next(f for f in folder.iterdir() if f.suffix == ".mcap")
        mcap_path = mcap_file
        hdf5_path = self.hdf5_dir / f"episode_{episode}.hdf5"
        for idx in self.indices:
            with self.subTest(idx=idx):
                mcap_data = self.extract_mcap_points(mcap_path, idx)
                hdf5_data = self.extract_hdf5_points(hdf5_path, idx)
                self.compare_data(mcap_data, hdf5_data)

    def extract_mcap_points(self, mcap_path, idx):
        piper_positions = []
        gripper_positions = []
        commands = []
        gripper_images = []
        main_images = []
        for msg in read_ros2_messages(str(mcap_path)):
            topic = msg.channel.topic
            ros_msg = msg.ros_msg
            if topic == "/left/piper/joint_states":
                piper_positions.append(np.array(ros_msg.position))  # 6 DoF
            elif topic == "/left/gripper_joint_state":
                gripper_positions.append(np.array(ros_msg.position))  # 1 DoF
            elif topic == "/left/leader_arm_control_node/commands":
                commands.append(np.array(ros_msg.position))  # 7 DoF
            elif topic == "/gripper/image_raw/compressed":
                gripper_images.append(decompress_image(ros_msg))
            elif topic == "/main/image_raw/compressed":
                main_images.append(decompress_image(ros_msg))
        # Combine qpos
        qpos_list = [
            np.concatenate([p, g]) for p, g in zip(piper_positions, gripper_positions)
        ]
        qpos = np.array(qpos_list)[idx]
        action = np.array(commands)[idx]
        gripper_img = gripper_images[idx]
        main_img = main_images[idx]
        return {
            "qpos": qpos,
            "action": action,
            "gripper_img": gripper_img,
            "main_img": main_img,
        }

    def extract_hdf5_points(self, hdf5_path, idx):
        with h5py.File(hdf5_path, "r") as f:
            qpos = f["observations/qpos"][idx]
            action = f["action"][idx]
            gripper_img = f["observations/images/gripper"][idx]
            main_img = f["observations/images/main"][idx]
        return {
            "qpos": qpos,
            "action": action,
            "gripper_img": gripper_img,
            "main_img": main_img,
        }

    def compare_data(self, mcap_data, hdf5_data):
        # Qpos
        self.assertTrue(
            np.allclose(mcap_data["qpos"], hdf5_data["qpos"], atol=1e-6),
            f"Qpos mismatch at idx: mcap {mcap_data['qpos']}, hdf5 {hdf5_data['qpos']}",
        )
        # Action
        self.assertTrue(
            np.allclose(mcap_data["action"], hdf5_data["action"], atol=1e-6),
            f"Action mismatch at idx: mcap {mcap_data['action']}, hdf5 {hdf5_data['action']}",
        )
        # Gripper img
        self.assertTrue(
            np.array_equal(mcap_data["gripper_img"], hdf5_data["gripper_img"]),
            "Gripper image mismatch",
        )
        # Main img
        self.assertTrue(
            np.array_equal(mcap_data["main_img"], hdf5_data["main_img"]),
            "Main image mismatch",
        )


if __name__ == "__main__":
    unittest.main()
