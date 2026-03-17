#!/usr/bin/env python
"""
后修复 LeRobot 数据集中的 task 名称。

用法:
1. 只重命名某个旧 task:
python shuol_scripts/fix_lerobot_task.py \
    --dataset_dir /path/to/dataset \
    --old_task "wrong task" \
    --new_task "correct task"

2. 将整个数据集强制统一成一个新 task:
python shuol_scripts/fix_lerobot_task.py \
    --dataset_dir /path/to/dataset \
    --new_task "correct task"
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="后修复 LeRobot 数据集中的 task 名称")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="LeRobot 数据集根目录",
    )
    parser.add_argument(
        "--new_task",
        type=str,
        required=True,
        help="修复后的 task 名称",
    )
    parser.add_argument(
        "--old_task",
        type=str,
        default=None,
        help="仅替换这个旧 task；不传时会把整个数据集统一成 new_task",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印将要执行的修改，不写回文件",
    )
    return parser.parse_args()


def ordered_unique(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def normalize_episode_tasks(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def rename_task_name(task_name: str, old_task: str | None, new_task: str) -> str:
    if old_task is None:
        return new_task
    return new_task if task_name == old_task else task_name


def load_info(info_path: Path) -> dict:
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_info(info: dict, info_path: Path) -> None:
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
        f.write("\n")


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    info_path = dataset_dir / "meta" / "info.json"
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    data_paths = sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet"))
    episode_paths = sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))

    if not dataset_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")
    if not info_path.exists():
        raise FileNotFoundError(f"缺少 info.json: {info_path}")
    if not tasks_path.exists():
        raise FileNotFoundError(f"缺少 tasks.parquet: {tasks_path}")
    if not data_paths:
        raise FileNotFoundError(f"未找到数据文件: {dataset_dir / 'data'}")
    if not episode_paths:
        raise FileNotFoundError(f"未找到 episode 元数据文件: {dataset_dir / 'meta' / 'episodes'}")

    tasks_df = pd.read_parquet(tasks_path)
    if "task_index" not in tasks_df.columns:
        raise ValueError(f"tasks.parquet 缺少 task_index 列: {tasks_path}")

    old_task_entries = [
        (str(task_name), int(task_index))
        for task_name, task_index in sorted(tasks_df["task_index"].items(), key=lambda item: item[1])
    ]
    old_tasks_by_index = [task_name for task_name, _ in old_task_entries]

    if args.old_task is not None and args.old_task not in old_tasks_by_index:
        raise ValueError(
            f"old_task 不在当前数据集中: {args.old_task}. 当前 tasks: {old_tasks_by_index}"
        )

    renamed_tasks_by_index = [
        rename_task_name(task_name, args.old_task, args.new_task)
        for task_name in old_tasks_by_index
    ]
    final_tasks = ordered_unique(renamed_tasks_by_index)
    old_index_to_new_index = {}
    for (_, old_task_index), renamed_task in zip(old_task_entries, renamed_tasks_by_index, strict=True):
        old_index_to_new_index[old_task_index] = final_tasks.index(renamed_task)

    print(f"数据集目录: {dataset_dir}")
    print(f"当前 tasks: {old_tasks_by_index}")
    if args.old_task is None:
        print(f"模式: 将整个数据集统一改为 `{args.new_task}`")
    else:
        print(f"模式: 将 `{args.old_task}` 重命名为 `{args.new_task}`")
    print(f"修改后 tasks: {final_tasks}")
    print(f"将修改 {len(data_paths)} 个 data parquet 和 {len(episode_paths)} 个 episode parquet")

    if args.dry_run:
        print("dry-run 模式，不写回任何文件。")
        return

    new_tasks_df = pd.DataFrame({"task_index": np.arange(len(final_tasks), dtype=np.int64)}, index=final_tasks)
    new_tasks_df.to_parquet(tasks_path)

    for data_path in data_paths:
        data_df = pd.read_parquet(data_path)
        if "task_index" not in data_df.columns:
            raise ValueError(f"data parquet 缺少 task_index 列: {data_path}")

        data_df["task_index"] = data_df["task_index"].map(old_index_to_new_index)
        if data_df["task_index"].isnull().any():
            raise ValueError(f"存在未映射的 task_index: {data_path}")
        data_df["task_index"] = data_df["task_index"].astype(np.int64)
        data_df.to_parquet(data_path, index=False)

    for episode_path in episode_paths:
        episode_df = pd.read_parquet(episode_path)
        if "tasks" not in episode_df.columns:
            raise ValueError(f"episode parquet 缺少 tasks 列: {episode_path}")

        episode_df["tasks"] = episode_df["tasks"].apply(
            lambda tasks: ordered_unique(
                rename_task_name(task_name, args.old_task, args.new_task)
                for task_name in normalize_episode_tasks(tasks)
            )
        )
        episode_df.to_parquet(episode_path, index=False)

    info = load_info(info_path)
    info["total_tasks"] = len(final_tasks)
    save_info(info, info_path)

    print("task 修复完成。")
    print(f"新的 tasks: {final_tasks}")


if __name__ == "__main__":
    main()
