#!/usr/bin/env python
"""
Convert Calvin.h5 dataset to LeRobot format

usage:
python calvin_to_lerobot.py --data_dir <data_dir> --output_dir <output_dir>
"""

import argparse
import json
import h5py
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 统一后的目标图像大小 / 形状
TARGET_IMAGE_SIZE = (256, 256)  # (width, height) for PIL.Image
TARGET_IMAGE_SHAPE = (256, 256, 3)  # (H, W, C) for features


def load_calvin_h5_file(h5_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a single h5 file
    
    Args:
        h5_path: h5 file path
        subdir_name: subdirectory name (for identifying trajectory source)
        
    Returns:
        trajectories list
    """

    trajectories = []

    
    trajectory_data = h5py.File(h5_path, "r")
    
    data_group = trajectory_data['data']
    # Get all trajectory keys, sorted by number
    demo_keys = sorted(
        data_group.keys(),
        key=lambda x: int(x.split("_")[1]) if "_" in x else x
    )

    for demo_key in demo_keys:
        traj_group = data_group[demo_key]
        
        demo_attr = traj_group.attrs

        # skip other behavior
        if "behavior" in demo_attr and demo_attr["behavior"] == "other":
            continue
        
        # 提取trajectory数据
        traj_data = {
            'task': demo_attr["behavior"],
        }
        
        # read actions
        if 'actions' in traj_group:
            traj_data['actions'] = traj_group['actions'][:]
        
        # 读取观测数据  
        if 'obs' in traj_group:
            obs_group = traj_group['obs']
            
            # 关节位置和速度
            if 'proprio' in obs_group:
                traj_data['qpos'] = obs_group['proprio'][:]
        
            # 基础相机
            if 'third_person' in obs_group:
                traj_data['third_person'] = obs_group['third_person'][:]
            
            # 手部相机
            if 'eye_in_hand' in obs_group:
                traj_data['eye_in_hand'] = obs_group['eye_in_hand'][:]
    
        trajectories.append(traj_data)
    
    trajectory_data.close()
    


    return trajectories

def _infer_features_from_h5(h5_path: str) -> Dict[str, Any]:
    """
    只读取一个trajectory的元信息来推断 LeRobotDataset 的 features，
    避免一次性把所有数据加载到内存。

    注意：我们会在转换时把所有相机图像统一 resize 到 TARGET_IMAGE_SHAPE，
    因此这里的 image feature shape 也直接写成 TARGET_IMAGE_SHAPE。
    """
    with h5py.File(h5_path, "r") as trajectory_data:
        data_group = trajectory_data["data"]
        demo_keys = sorted(
            data_group.keys(),
            key=lambda x: int(x.split("_")[1]) if "_" in x else x,
        )

        for demo_key in demo_keys:
            traj_group = data_group[demo_key]
            demo_attr = traj_group.attrs

            # 跳过 other 行为
            if "behavior" in demo_attr and demo_attr["behavior"] == "other":
                continue

            features: Dict[str, Any] = {}

            # 观测：proprio
            if "obs" in traj_group and "proprio" in traj_group["obs"]:
                proprio_ds = traj_group["obs"]["proprio"]
                proprio_shape = proprio_ds.shape[1:]
                features["observation.state"] = {
                    "dtype": "float32",
                    "shape": proprio_shape,
                    "names": [f"proprio_{i}" for i in range(proprio_shape[0])],
                }
                print(f"  ✓ observation.state (proprio): {proprio_shape}")

            # 动作
            if "actions" in traj_group:
                action_ds = traj_group["actions"]
                action_shape = action_ds.shape[1:]
                features["action"] = {
                    "dtype": "float32",
                    "shape": action_shape,
                    "names": [f"action_{i}" for i in range(action_shape[0])],
                }
                print(f"  ✓ action: {action_shape}")

            # 基础相机
            if "obs" in traj_group and "third_person" in traj_group["obs"]:
                img_ds = traj_group["obs"]["third_person"]
                orig_img_shape = img_ds.shape[1:]  # (H, W, C)
                features["observation.images.third_person"] = {
                    "dtype": "video",
                    "shape": TARGET_IMAGE_SHAPE,
                    "names": ["height", "width", "channels"],
                }
                print(
                    f"  ✓ observation.images.third_person: {orig_img_shape} -> resized to {TARGET_IMAGE_SHAPE}"
                )

            # 手部相机
            if "obs" in traj_group and "eye_in_hand" in traj_group["obs"]:
                img_ds = traj_group["obs"]["eye_in_hand"]
                orig_img_shape = img_ds.shape[1:]  # (H, W, C)
                features["observation.images.eye_in_hand"] = {
                    "dtype": "video",
                    "shape": TARGET_IMAGE_SHAPE,
                    "names": ["height", "width", "channels"],
                }
                print(
                    f"  ✓ observation.images.eye_in_hand: {orig_img_shape} -> resized to {TARGET_IMAGE_SHAPE}"
                )

            return features

    return {}


def create_lerobot_dataset(args: argparse.Namespace):
    """
    流式地将 Calvin .h5 数据转换为 LeRobot 格式：
    - 先用一个 trajectory 推断特征结构
    - 再逐个 trajectory、逐帧读取并写入 LeRobotDataset
    """

    features = _infer_features_from_h5(args.h5_file)

    # 检查必需特征
    if "observation.state" not in features or "action" not in features:
        print("\n错误: 缺少必需的特征 (observation.state 或 action)")
        return

    task_output_dir = os.path.join(args.output_dir, "calvin_D_train_new")

    dataset = LeRobotDataset.create(
        repo_id="calvin_D_train_new",
        fps=args.fps,
        root=task_output_dir,
        features=features,
        use_videos=True,  # 使用视频编码节省空间
    )

    print("\n开始流式转换 trajectories...")

    # 处理每个 trajectory（不在内存中累积 list）
    import tqdm

    with h5py.File(args.h5_file, "r") as trajectory_data:
        data_group = trajectory_data["data"]
        demo_keys = sorted(
            data_group.keys(),
            key=lambda x: int(x.split("_")[1]) if "_" in x else x,
        )

        for traj_idx, demo_key in enumerate(
            tqdm.tqdm(demo_keys, desc="处理 trajectories")
        ):


            traj_group = data_group[demo_key]
            demo_attr = traj_group.attrs

            # 跳过 other 行为
            if "behavior" in demo_attr and demo_attr["behavior"] == "other":
                continue


            task_name = demo_attr.get("behavior", "unknown")

            # 数据集句柄（懒加载，按帧读取）
            actions_ds = traj_group.get("actions", None)
            obs_group = traj_group.get("obs", None)

            if obs_group is None or actions_ds is None:
                print(f"  警告: trajectory {demo_key} 缺少 obs 或 actions，跳过")
                continue

            proprio_ds = obs_group.get("proprio", None)
            third_person_ds = obs_group.get("third_person", None)
            eye_in_hand_ds = obs_group.get("eye_in_hand", None)

            if proprio_ds is None:
                print(f"  警告: trajectory {demo_key} 缺少 proprio，跳过")
                continue

            # 使用 actions 的长度作为 trajectory 长度
            traj_length = len(actions_ds)

            if traj_length > 100 or traj_length < 30:
                continue

            for frame_idx in range(traj_length):
                frame_data: Dict[str, Any] = {
                    "task": task_name,
                }

                # 观测状态
                frame_data["observation.state"] = np.asarray(
                    proprio_ds[frame_idx], dtype=np.float32
                )

                # 动作
                frame_data["action"] = np.asarray(
                    actions_ds[frame_idx], dtype=np.float32
                )

                # 基础相机
                if third_person_ds is not None and frame_idx < len(third_person_ds):
                    img_data = np.asarray(third_person_ds[frame_idx], dtype=np.uint8)
                    img = Image.fromarray(img_data)
                    if img.size != TARGET_IMAGE_SIZE:
                        img = img.resize(TARGET_IMAGE_SIZE, Image.Resampling.BILINEAR)
                    frame_data["observation.images.third_person"] = img

                # 手部相机
                if eye_in_hand_ds is not None and frame_idx < len(eye_in_hand_ds):
                    img_data = np.asarray(eye_in_hand_ds[frame_idx], dtype=np.uint8)
                    img = Image.fromarray(img_data)
                    if img.size != TARGET_IMAGE_SIZE:
                        img = img.resize(TARGET_IMAGE_SIZE, Image.Resampling.BILINEAR)
                    frame_data["observation.images.eye_in_hand"] = img

                # 添加帧
                try:
                    dataset.add_frame(frame_data)
                except Exception as e:
                    print(
                        f"  错误: trajectory {demo_key} frame {frame_idx} 添加失败: {e}"
                    )
                    continue

            # 一个 trajectory 结束，保存 episode
            try:
                dataset.save_episode()
            except Exception as e:
                print(f"  错误: 保存 trajectory {demo_key} 失败: {e}")
                continue

    print(f"\n数据集创建完成!")
    print(f"总episodes: {dataset.meta.total_episodes}")
    print(f"总frames: {dataset.meta.total_frames}")
    print(f"输出目录: {task_output_dir}")
    

def main():
    parser = argparse.ArgumentParser(description="将ManiSkill .h5数据转换为LeRobot格式")
    parser.add_argument("--h5_file", type=str,
                       default="/home/shuo/research/datasets/CALVIN_D_H5/CalvinD_train_betterseg/data.hdf5",
                       help="输入.h5文件")
    parser.add_argument("--output_dir", type=str,
                       default="/home/shuo/research/datasets/steerDP",
                       help="输出目录")
    parser.add_argument("--fps", type=int, default=10, help="数据采集帧率")
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.h5_file):
        print(f"错误: 输入目录不存在: {args.h5_file}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 直接流式转换为 LeRobot 格式（不一次性加载全部 trajectory）
    print("\n" + "=" * 80)
    print("转换为LeRobot格式...")
    print("=" * 80)
    create_lerobot_dataset(args)
    
    print("\n" + "=" * 80)
    print("转换完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
