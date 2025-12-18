import argparse
import os
import h5py
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages
import cv2

def decompress_image(msg):
    """Decompress CompressedImage to RGB numpy array."""
    fmt = msg.format.lower()
    if 'jpeg' in fmt:
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if 'bgr8' in fmt:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    elif 'png' in fmt:
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            img = img[:, :, :3]  # Drop alpha
        if 'bgr8' in fmt:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        raise ValueError(f"Unsupported image format: {msg.format}")

def process_trajectory(mcap_path, output_path, episode_id, print_dims=False):
    """Process a single .mcap file and save as HDF5."""
    data = {
        'qpos': [],
        'qvel': [],  # Will be zeros matching qpos shape
        'action': [],
        'gripper_image': [],
        'main_image': []
    }
    
    piper_positions = []
    gripper_positions = []
    commands = []
    gripper_images = []
    main_images = []
    
    for msg in read_ros2_messages(mcap_path):
            topic = msg.channel.topic
            ros_msg = msg.ros_msg
            
            if topic == '/left/piper/joint_states':
                piper_positions.append(np.array(ros_msg.position))  # 6 DoF
            elif topic == '/left/gripper_joint_state':
                gripper_positions.append(np.array(ros_msg.position))  # 1 DoF
            elif topic == '/left/leader_arm_control_node/commands':
                commands.append(np.array(ros_msg.position))  # 7 DoF
            elif topic == '/gripper/image_raw/compressed':
                gripper_images.append(decompress_image(ros_msg))
            elif topic == '/main/image_raw/compressed':
                main_images.append(decompress_image(ros_msg))
    
    # Combine qpos: piper (6) + gripper (1)
    if piper_positions and gripper_positions:
        qpos_list = [np.concatenate([p, g]) for p, g in zip(piper_positions, gripper_positions)]
        qpos = np.array(qpos_list)
        qvel = np.zeros_like(qpos)  # No velocities, set to zeros
    else:
        raise ValueError("Missing piper or gripper joint states")
    
    action = np.array(commands)
    gripper_images = np.array(gripper_images)
    main_images = np.array(main_images)
    
    if print_dims and episode_id == 0:
        print(f"Episode 0 dimensions:")
        print(f"  qpos: {qpos.shape}")
        print(f"  qvel: {qvel.shape}")
        print(f"  action: {action.shape}")
        print(f"  gripper_image: {gripper_images.shape}")
        print(f"  main_image: {main_images.shape}")
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        f.attrs['sim'] = False
        f.create_dataset('/action', data=action)
        obs_group = f.create_group('/observations')
        obs_group.create_dataset('qpos', data=qpos)
        obs_group.create_dataset('qvel', data=qvel)
        images_group = obs_group.create_group('images')
        images_group.create_dataset('gripper', data=gripper_images)
        images_group.create_dataset('main', data=main_images)
    
    print(f"Saved episode {episode_id} to {output_path}")

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    episode_id = 0
    for folder in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            mcap_file = None
            for file in os.listdir(folder_path):
                if file.endswith('.mcap'):
                    mcap_file = os.path.join(folder_path, file)
                    break
            if mcap_file:
                output_path = os.path.join(output_dir, f'episode_{episode_id}.hdf5')
                process_trajectory(mcap_file, output_path, episode_id, print_dims=True if episode_id == 0 else False)
                episode_id += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/home/eai/data/fruits_picking')
    parser.add_argument('--output_dir', type=str, default='/home/eai/act/data/hdf5_episodes')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)