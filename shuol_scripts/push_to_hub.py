from huggingface_hub import HfApi

# 配置
repo_id = "TreeePlanter/vls_maniskill_cubes"
model_path = "/home/shuo/research/lerobot/outputs/train/2026-01-22/18-12-45_vls_real_dp_l2/checkpoints/100000/pretrained_model"
private = False

api = HfApi()

# 创建 repo（如果不存在）
api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

# 上传文件夹
commit_info = api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=model_path,
    commit_message="Upload policy weights and config",
    allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
    ignore_patterns=["*.tmp", "*.log"],
)

print(f"Model pushed to {commit_info.repo_url}")

