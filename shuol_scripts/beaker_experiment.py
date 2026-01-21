"""
简化版 Beaker 示例：把常用步骤直接放到一个示例函数中，移除不必要的轻量封装函数。

保留必要的导入与最小流程：
- 初始化 Beaker 客户端
- 从 YAML 加载 spec
- 直接在示例中配置 task（资源、镜像、挂载、环境、命令）
- 创建 1 个或多个 experiment 并将它们放到 group

这个文件只做示例说明，便于快速拷贝到脚本中使用。
"""

import copy
import base64
import json

from beaker import Beaker, BeakerEnvVar, BeakerExperimentSpec



def example_full_workflow():
    """简洁的端到端示例：初始化 → 加载 spec → 修改 task → 创建 experiments → 建 group"""
    print("=" * 80)
    print("Beaker experiment for lerobot training")
    print("=" * 80)

    # 初始化 Beaker（从环境变量读取 BEAKER_TOKEN）
    beaker = Beaker.from_env(default_workspace="ai2/shuol")
    print(f"Beaker user: {beaker.user_name}")

    # 从 YAML 加载基础 spec
    yaml_path = "shuol_scripts/beaker_train.yml"
    spec = BeakerExperimentSpec.from_file(yaml_path)
    print(f"Loaded spec with {len(spec.tasks)} task(s)")

    # 直接配置第一个 task（示例中只修改常见字段）
    task = spec.tasks[0]

    # 数据挂载示例（直接在这里追加）
    if task.datasets is None:
        task.datasets = []
    
    task.arguments = [
        "python /weka/prior/shuol/lerobot/src/lerobot/scripts/lerobot_train.py "
        "--dataset.repo_id=TreeePlanter/vls_real_l1_red_mug_handle "
        "--dataset.repo_ids=\"['TreeePlanter/vls_real_l1_red_mug_handle','TreeePlanter/vls_real_l1_red_mug_rim','TreeePlanter/vls_real_l2_orange_red','TreeePlanter/vls_real_l2_banana_red','TreeePlanter/vls_real_l1_orange_red','TreeePlanter/vls_real_l2_orange_green','TreeePlanter/vls_real_l1_orange_green','TreeePlanter/vls_real_l2_banana_green']\" "
        "--policy.type=diffusion "
        "--wandb.enable=true "
        "--output_dir=/weka/prior/shuol/lerobot/outputs"
    ]

    # 创建多个 experiments（如果只需一个，将 count 设为 1）
    import shortuuid
    count = 1
    run_uuid = str(shortuuid.uuid())
    experiments = []
    for i in range(count):
        s = copy.deepcopy(spec)
        s.name = f"vls_lerobot__run_{i+1}__{run_uuid}"
        s.description = f"Run {i+1} of {count}"
        print(f"Creating experiment: {s.name}")
        exp = beaker.experiment.create(spec=s, workspace="ai2/shuol")
        experiments.append(exp)
        print(f"  -> id: {exp.experiment.id}")

    # 把 experiments 放到一个 group（可选）
    # experiment_ids = [e.experiment.id for e in experiments]
    # group = beaker.group.create("vls_real", experiment_ids=experiment_ids, description="vls_realworld_training")
    # print(f"Group created: {group.full_name} ({len(experiment_ids)} experiments)")

if __name__ == "__main__":
    example_full_workflow()
