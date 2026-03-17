from pathlib import Path
from huggingface_hub import HfApi
from lerobot.datasets.utils import create_lerobot_dataset_card, load_info

# 配置
repo_id = "TreeePlanter/molmoact2_simbench_droid_open_drawer_pnp_fork_replay"  # 修改为你的 repo id
dataset_path = "/home/shuo/research/datasets/molmoact2/sim_bench/droid/maniskill_DroidKitchenOpenDrawerPnpFork_replay"  # 修改为你的数据集路径
private = False

# LeRobot 数据集版本（必须与 lerobot 代码库版本匹配）
CODEBASE_VERSION = "v3.0"

api = HfApi()

# 创建 repo（如果不存在）
api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="dataset")

# 上传文件夹
commit_info = api.upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=dataset_path,
    commit_message="Upload dataset",
    ignore_patterns=["*.tmp", "*.log", "__pycache__", ".git"],
)

# 创建并上传 dataset card（添加 Tags: LeRobot, Tasks: Robotics）
dataset_info = load_info(Path(dataset_path))
card = create_lerobot_dataset_card(
    tags=None,  # 可以添加额外标签，如 ["simulation", "maniskill"]
    dataset_info=dataset_info,
    license="apache-2.0",
)
card.push_to_hub(repo_id=repo_id, repo_type="dataset")

# 创建版本标签（LeRobot 必须）
try:
    api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
except Exception:
    pass  # 标签不存在，忽略
api.create_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")

print(f"Dataset pushed to {commit_info.repo_url}")
print(f"Version tag '{CODEBASE_VERSION}' created.")
print("Dataset card with 'LeRobot' tag uploaded.")

