# Roban2 紧凑动作库训练

此路径沿用现有 Roban2 SuperMimic 观测、奖励、世界模型和策略，仅替换参考动作存储：每帧保留 50 Hz、59 维 FP16 根部与关节状态，参考刚体状态在用到时由 Roban2 正运动学计算。现有逐条 `.npz` 配置仍可使用。

## 构建动作库

数据集根目录必须包含 `passed_motions.json`。默认打包全部质检通过的轨迹；名称采用相对路径而非文件名，避免不同类别下的同名轨迹冲突。本地清单中有两处目录名带尾部句点而实际文件夹没有，打包工具会打印修正记录；其他缺失文件仍会报错。先用少量轨迹检查环境：

```bash
uv run python -m tools.build_roban2_compact_library \
  --dataset-root /home/lyx/train/dataset/LET-HumanMotion-Roban2 \
  --output /home/lyx/train/dataset/roban2_smoke_compact.pt --limit 20
```

正式训练建议先覆盖没有额外道具和场景交互的轨迹。以下命令包含 `Non-Interactive_Actions`、`Dance` 和 `Wushu` 三类，并只使用质检通过的文件：

```bash
uv run python -m tools.build_roban2_compact_library \
  --dataset-root /home/lyx/train/dataset/LET-HumanMotion-Roban2 \
  --include-prefix Non-Interactive_Actions \
  --include-prefix Dance --include-prefix Wushu \
  --output /home/lyx/train/dataset/roban2_body_motion_compact.pt
```

省略 `--include-prefix` 会打包质检通过的全部数据，但当前平地、无道具场景不能正确评测物体和环境交互任务。库文件约按轨迹总时长线性增长；全部 204 小时的参考状态约占 4 GiB GPU 显存，另需帧索引、仿真环境和训练缓冲区。打包过程会读取源轨迹两遍，并在输出目录使用临时映射文件；完成后自动删除临时文件。

## 启动训练

```bash
export ROBAN2_COMPACT_LIBRARY=/home/lyx/train/dataset/roban2_body_motion_compact.pt
uv run unisupermimic-train task=roban2_super_mimic/compact_passed \
  training.log_dir=runs/roban2_compact_full
```

`compact_passed` 默认 2048 环境；24 GB RTX 4090 上先用小库做短轮次显存和速度测试，再增加轨迹数。若显存仍不足，降低 `algo.num_envs` 或 `algo.pi_batch`。紧凑库会增加在线正运动学计算，不能从 mixed-370 的每秒迭代数直接推算全量训练时间；12000 轮也不足以保证约数万条轨迹获得与 370 条实验相同的采样覆盖。

全量评测不能一次为数万条轨迹各开一个环境。可将固定留出轨迹写成 `{"passed": ["相对路径/clip.npz", ...]}` 的 JSON 清单，用 `--manifest` 构建评测子库，再按类别分批评测。训练和留出清单应先按相对路径去重并保持不交叉。旧的 gait-120、mixed-370 checkpoint 与本配置的动作库清单不同，不应直接作为严格续训使用；新实验从头训练并单独保存运行目录。
