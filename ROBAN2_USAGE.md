# Roban2 SuperMimic 使用说明

该任务使用 OpenLET `LET-HumanMotion-Roban2` 自带的 23 关节 MuJoCo 模型。它与 G1 任务并存；LejuLab-Deploy 的 21 关节模型不在此训练配置中。

1. 将数据集放在 `/home/lyx/train/dataset/LET-HumanMotion-Roban2`，或设置 `ROBAN2_DATASET_ROOT` 指向数据集根目录。先执行 `uv run tools/prepare_roban2_assets.py --source <数据集根目录>/scripts/robot`，生成被 `.gitignore` 排除的本地机器人场景及 mesh。数据集许可见其 `README.md`（CC BY-NC-SA 4.0）。
2. 默认 Hydra 配置只使用一条 NPZ 作小规模验证。换片段时覆盖 `env.commands.motion.motion_files`，列表元素是 NPZ 的完整路径；不要直接加载约 4 万条全量轨迹，当前运动库会预计算并常驻全部帧和跟踪 body 的状态。
3. 训练入口的任务选择为 `task=roban2_super_mimic/mujoco` 或 `task=roban2_super_mimic/mjwarp`。例如：`uv run unisupermimic-train task=roban2_super_mimic/mujoco algo.num_envs=1 algo.compile=false`。本文档的命令是后续运行入口，本次移植未启动训练。

NPZ 的 `qpos(T,30)` 和 `data(T,30)` 直接匹配数据集模型的 MuJoCo 关节顺序；`qpos(T,28)` 在末尾补两个头部关节零位。根朝向四元数保持 MuJoCo 的 `wxyz` 顺序。采样率由 120 Hz 插值到本任务的 50 Hz 控制周期。训练动作使用由数据集 XML 的 torque motor 转换得到的位置 PD actuator；身体 21 个关节采用 LejuLab-Deploy mimic 配置中的 `kp/kd`，头部 2 个关节使用仅供仿真的默认值。训练出的 23 关节策略不能直接部署到 21 关节控制器。
