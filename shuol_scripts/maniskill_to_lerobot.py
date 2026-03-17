#!/usr/bin/env python
"""
将 ManiSkill .h5 数据集转换为 LeRobot 格式

用法:
python maniskill_to_lerobot.py --h5_file <h5_file_path> [<h5_file_path> ...] --output_dir <output_dir> [--dataset_name <name>]
"""

import argparse
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
TARGET_IMAGE_SIZE = (640, 480)  # (width, height) for PIL.Image
TARGET_IMAGE_SHAPE = (480, 640, 3)  # (H, W, C) for features
CAMERA_MAPPING = {
    "yam":{
        "base_camera": "base_camera",
        "left_hand_camera": "left_hand_camera",
        "right_hand_camera": "right_hand_camera",
    },
    "droid":{
        "ex_right_camera": "ex_right_camera",
        "hand_camera": "hand_camera",
        "ex_left_camera": "ex_left_camera",
        # "ex_right_camera": "ex_right_camera",
    }
}


def _get_camera_mapping(robot_type: str) -> Dict[str, str]:
    if robot_type not in CAMERA_MAPPING:
        raise ValueError(
            f"不支持的 robot_type: {robot_type}. 可选值: {sorted(CAMERA_MAPPING.keys())}"
        )
    return CAMERA_MAPPING[robot_type]


def _infer_features_from_h5(h5_path: str, robot_type: str) -> Dict[str, Any]:
    """
    只读取一个trajectory的元信息来推断 LeRobotDataset 的 features，
    避免一次性把所有数据加载到内存。

    注意：我们会在转换时把所有相机图像统一 resize 到 TARGET_IMAGE_SHAPE，
    因此这里的 image feature shape 也直接写成 TARGET_IMAGE_SHAPE。
    """
    camera_mapping = _get_camera_mapping(robot_type)

    with h5py.File(h5_path, "r") as f:
        # ManiSkill h5 文件的轨迹键名格式: traj_0, traj_1, ...
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith('traj_')],
            key=lambda x: int(x.split('_')[1])
        )

        for traj_key in traj_keys:
            traj_group = f[traj_key]

            features: Dict[str, Any] = {}

            # 检查 obs 组
            if 'obs' not in traj_group:
                continue

            obs_group = traj_group['obs']

            # 1. 处理 agent 状态 (qpos, qvel)
            if 'agent' in obs_group:
                agent_group = obs_group['agent']
                
                # qpos
                if 'qpos' in agent_group:
                    qpos_ds = agent_group['qpos']
                    qpos_shape = qpos_ds.shape[1:]  # (时间步, 关节数) -> (关节数,)
                    
                    state_dim = qpos_shape[0]
                    features["observation.state"] = {
                        "dtype": "float32",
                        "shape": (state_dim,),
                        "names": [f"qpos_{i}" for i in range(qpos_shape[0])] \
                    }
                    print(f"  ✓ observation.state: qpos{qpos_shape} = {state_dim}")

            # 2. 处理 extra 组 (tcp_pose)
            # if 'extra' in obs_group and 'tcp_pose' in obs_group['extra']:
            #     tcp_pose_ds = obs_group['extra']['tcp_pose']
            #     tcp_pose_shape = tcp_pose_ds.shape[1:]  # (时间步, 7) -> (7,)
            #     features["observation.tcp_pose"] = {
            #         "dtype": "float32",
            #         "shape": tcp_pose_shape,
            #         "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
            #     }
            #     print(f"  ✓ observation.tcp_pose: {tcp_pose_shape}")

            # 3. 处理动作
            if 'actions' in traj_group:
                action_ds = traj_group['actions']
                action_shape = action_ds.shape[1:]  # (时间步, 动作维度) -> (动作维度,)
                features["action"] = {
                    "dtype": "float32",
                    "shape": action_shape,
                    "names": [f"action_{i}" for i in range(action_shape[0])],
                }
                print(f"  ✓ action: {action_shape}")

            # 4. 处理相机图像
            if 'sensor_data' in obs_group:
                sensor_data_group = obs_group['sensor_data']

                for ms_cam_name, lerobot_cam_name in camera_mapping.items():
                    if ms_cam_name in sensor_data_group:
                        cam_group = sensor_data_group[ms_cam_name]
                        if 'rgb' in cam_group:
                            img_ds = cam_group['rgb']
                            orig_img_shape = img_ds.shape[1:]  # (T, H, W, C) -> (H, W, C)
                            features[f"observation.images.{lerobot_cam_name}"] = {
                                "dtype": "video",
                                "shape": TARGET_IMAGE_SHAPE,
                                "names": ["height", "width", "channels"],
                            }
                            print(
                                f"  ✓ observation.images.{lerobot_cam_name}: {orig_img_shape} -> resized to {TARGET_IMAGE_SHAPE}"
                            )

            return features

    return {}


def _validate_features_compatible(
    h5_paths: List[str], base_features: Dict[str, Any], robot_type: str
) -> bool:
    """确保所有 h5 文件的特征结构一致。"""
    for h5_path in h5_paths[1:]:
        current_features = _infer_features_from_h5(h5_path, robot_type)
        if current_features != base_features:
            print(f"\n错误: h5 文件特征结构不一致: {h5_path}")
            print(f"基准特征键: {sorted(base_features.keys())}")
            print(f"当前特征键: {sorted(current_features.keys())}")
            return False
    return True


def _resolve_dataset_name(h5_paths: List[str], explicit_name: str | None) -> str:
    """为单文件或多文件输入生成统一的数据集名称。"""
    if explicit_name:
        return explicit_name

    if len(h5_paths) == 1:
        return Path(h5_paths[0]).stem

    parent_name = Path(h5_paths[0]).parent.name
    return parent_name or "combined_h5_dataset"


def create_lerobot_dataset(args: argparse.Namespace):
    """
    流式地将 ManiSkill .h5 数据转换为 LeRobot 格式：
    - 先用一个 trajectory 推断特征结构
    - 再逐个 trajectory、逐帧读取并写入 LeRobotDataset
    """

    h5_paths = args.h5_file
    camera_mapping = _get_camera_mapping(args.robot_type)
    features = _infer_features_from_h5(h5_paths[0], args.robot_type)

    # 检查必需特征
    if "observation.state" not in features or "action" not in features:
        print("\n错误: 缺少必需的特征 (observation.state 或 action)")
        return

    if not _validate_features_compatible(h5_paths, features, args.robot_type):
        return

    dataset_name = _resolve_dataset_name(h5_paths, args.dataset_name)
    task_output_dir = os.path.join(args.output_dir, f"maniskill_{dataset_name}")

    dataset = LeRobotDataset.create(
        repo_id=f"maniskill_{dataset_name}",
        fps=args.fps,
        root=task_output_dir,
        features=features,
        use_videos=True,  # 使用视频编码节省空间
    )

    print(f"\n开始流式转换 trajectories... 共 {len(h5_paths)} 个 h5 文件")

    # 处理每个 trajectory（不在内存中累积 list）
    import tqdm

    for h5_path in h5_paths:
        print(f"\n处理 h5 文件: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            # 获取所有 trajectory 键
            traj_keys = sorted(
                [k for k in f.keys() if k.startswith('traj_')],
                key=lambda x: int(x.split('_')[1])
            )

            for traj_idx, traj_key in enumerate(
                tqdm.tqdm(traj_keys, desc=f"处理 trajectories ({Path(h5_path).name})")
            ):
                traj_group = f[traj_key]

                # 数据集句柄（懒加载，按帧读取）
                actions_ds = traj_group.get("actions", None)
                obs_group = traj_group.get("obs", None)

                if obs_group is None or actions_ds is None:
                    print(f"  警告: trajectory {traj_key} 缺少 obs 或 actions，跳过")
                    continue

                # 获取 agent 数据
                agent_group = obs_group.get("agent", None)
                if agent_group is None:
                    print(f"  警告: trajectory {traj_key} 缺少 agent 数据，跳过")
                    continue

                qpos_ds = agent_group.get("qpos", None)
                # qvel_ds = agent_group.get("qvel", None)

                if qpos_ds is None:
                    print(f"  警告: trajectory {traj_key} 缺少 qpos，跳过")
                    continue

                # 获取 extra 数据
                # extra_group = obs_group.get("extra", None)
                # tcp_pose_ds = None
                # if extra_group is not None:
                #     tcp_pose_ds = extra_group.get("tcp_pose", None)

                # 获取相机数据
                sensor_data_group = obs_group.get("sensor_data", None)
                camera_data = {}
                if sensor_data_group is not None:
                    for ms_cam_name, lerobot_cam_name in camera_mapping.items():
                        if ms_cam_name in sensor_data_group:
                            cam_group = sensor_data_group[ms_cam_name]
                            if 'rgb' in cam_group:
                                camera_data[lerobot_cam_name] = cam_group['rgb']

                # 使用 actions 的长度作为 trajectory 长度
                traj_length = len(actions_ds)

                # 过滤轨迹长度（可选）
                if args.min_length is not None and traj_length < args.min_length:
                    continue
                if args.max_length is not None and traj_length > args.max_length:
                    continue

                for frame_idx in range(traj_length):
                    frame_data: Dict[str, Any] = {
                        "task": args.task_name,  # 使用文件名作为任务名
                    }

                    # 观测状态: 合并 qpos 和 qvel
                    # qpos = np.asarray(qpos_ds[frame_idx], dtype=np.float32)
                    # qvel = np.asarray(qvel_ds[frame_idx], dtype=np.float32)
                    frame_data["observation.state"] = np.asarray(qpos_ds[frame_idx], dtype=np.float32)

                    # TCP 位姿
                    # if tcp_pose_ds is not None and frame_idx < len(tcp_pose_ds):
                    #     frame_data["observation.tcp_pose"] = np.asarray(
                    #         tcp_pose_ds[frame_idx], dtype=np.float32
                    #     )

                    # 动作
                    frame_data["action"] = np.asarray(
                        actions_ds[frame_idx], dtype=np.float32
                    )

                    # 相机图像
                    for lerobot_cam_name, cam_ds in camera_data.items():
                        if frame_idx < len(cam_ds):
                            img_data = np.asarray(cam_ds[frame_idx], dtype=np.uint8)
                            img = Image.fromarray(img_data)
                            if img.size != TARGET_IMAGE_SIZE:
                                img = img.resize(TARGET_IMAGE_SIZE, Image.Resampling.BILINEAR)
                            frame_data[f"observation.images.{lerobot_cam_name}"] = img

                    # 添加帧
                    try:
                        dataset.add_frame(frame_data)
                    except Exception as e:
                        print(
                            f"  错误: 文件 {Path(h5_path).name} trajectory {traj_key} frame {frame_idx} 添加失败: {e}"
                        )
                        continue

                # 一个 trajectory 结束，保存 episode
                try:
                    dataset.save_episode()
                except Exception as e:
                    print(f"  错误: 保存文件 {Path(h5_path).name} 中的 trajectory {traj_key} 失败: {e}")
                    continue

    print(f"\n数据集创建完成!")
    print(f"总episodes: {dataset.meta.total_episodes}")
    print(f"总frames: {dataset.meta.total_frames}")
    print(f"输出目录: {task_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="将 ManiSkill .h5 数据转换为 LeRobot 格式")
    parser.add_argument(
        "--h5_file",
        type=str,
        nargs="+",
        default=[
            "/home/shuo/research/molmospaces/molmo_spaces_maniskill/demos/DroidKitchenOpenDrawerPnpFork-v1/gello_teleop_20260310_160343/replay/trajectory.rgb.pd_joint_pos.physx_cpu.h5",
            
        
        ],
        help="输入一个或多个 .h5 文件路径"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="DroidKitchenOpenDrawerPnpFork_replay",
        help="输出数据集名称；多 h5 合并时建议显式传入"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/shuo/research/datasets/molmoact2/sim_bench/droid/",
        help="输出目录"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="数据采集帧率"
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        choices=sorted(CAMERA_MAPPING.keys()),
        default="droid",
        help="选择使用哪套相机映射"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="open top drawer and put the fork in the pan",
        help="任务名称"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=None,
        help="最小轨迹长度（可选过滤）"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="最大轨迹长度（可选过滤）"
    )

    args = parser.parse_args()

    # 检查输入文件
    missing_h5_files = [h5_path for h5_path in args.h5_file if not os.path.exists(h5_path)]
    if missing_h5_files:
        print("错误: 以下输入文件不存在:")
        for h5_path in missing_h5_files:
            print(f"  - {h5_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 直接流式转换为 LeRobot 格式（不一次性加载全部 trajectory）
    print("\n" + "=" * 80)
    print("转换为 LeRobot 格式...")
    print("=" * 80)
    create_lerobot_dataset(args)

    print("\n" + "=" * 80)
    print("转换完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()

