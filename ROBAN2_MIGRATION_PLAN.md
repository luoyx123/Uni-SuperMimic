# Roban2 迁移计划与可行性评估

## 目标与结论

目标是在本仓库的 UniLab/UniSim + SuperMimic 训练流程中，用 Roban2（`biped_s17`）和 OpenLET `LET-HumanMotion-Roban2` 轨迹训练运动模仿策略。第一阶段只要求仿真中训练和评估；真机部署是独立的后续阶段，不能把训练成功等同于可以直接部署。

**结论：可以迁移，但不是换一个 XML 和数据路径的配置工作。建议先做 1~2 天的最小闭环验证，再决定是否继续。** 第一版训练优先采用数据集自带的 23 关节 MJCF，与多数 NPZ 的 `qpos` 顺序一致；不需要 URDF。算法的世界模型、策略和回放缓存大体按任务提供的关节数构建；G1 任务的运动读取、正运动学、观测/奖励、动作、随机化和资产校验则写死了 G1 结构。若目标是尽快使用整套 OpenLET 数据并接入 Leju 原有部署控制器，应同时比较已有的 Leju/Isaac Lab 训练流程。

## 已核实的接口与差异

| 项目 | 当前仓库 G1 | Roban2 资源 | 影响 |
|---|---|---|---|
| 机器人 | 29 个可控关节，根 body 为 `pelvis` | 数据集 MJCF：23 个 actuator，`nq=30`，根 body 为 `base_link`；部署 MJCF：21 个 actuator，`nq=28` | 场景实体、动作、跟踪 body 和状态维度都需改变；训练与部署模型规格不同 |
| 轨迹模型 | G1 的 29 关节 CSV，根四元数按 `xyzw` 读取 | 数据集自带 `biped_s17.xml` 的关节声明顺序与 30 列 NPZ 一致，含 2 个头部关节 | 可直接在 MuJoCo 中回放 30 列轨迹，无需 URDF；当前 G1 任务仍不能直接使用 |
| 轨迹 | 40 个 CSV，36 列，30 fps，启动时整库载入并预计算 FK | NPZ：`qpos` 或 `data`；宽度 30 或 28；120 fps，根四元数 `wxyz` | 需格式适配、插值/降采样和内存策略 |
| 关节顺序 | G1 自然关节顺序 | 数据集 XML/30 列数据按腰、左腿、右腿、左臂、右臂、头；ROS/部署 XML 中腿在腰前；部署 YAML 以腰开头 | 更换模型或导出策略时必须按**关节名**映射，不能直接切片对齐 |
| 控制 | G1 使用位置 actuator；目标为默认姿态 + 随机偏置 + 缩放动作 | Leju XML 的 actuator 为 `motor`；部署 YAML 同时包含力矩/位置控制模式，并标记 `residual_action: true` | 要定义训练时 PD、力矩限幅和部署侧动作语义，不能直接复用 G1 增益 |
| 训练时钟 | `sim_dt=0.005`、`ctrl_dt=0.02` | 部署示例 `policy_dt=0.02`；数据 120 fps | 控制频率可对齐，但需做时间戳插值 |

本地核对路径：

- 本仓库任务代码：`src/unisupermimic/tasks/motion_tracking/g1/supermimic/`，配置：`src/unisupermimic/conf/supermimic/`。
- 轨迹：`/home/lyx/train/dataset/LET-HumanMotion-Roban2/`；`/train/dataset` 在当前机器上不存在。数据集仍在 clone，以完成后的文件数和校验结果为准。
- 数据集模型：`/home/lyx/train/dataset/LET-HumanMotion-Roban2/scripts/robot/biped_s17.xml`。
- ROS 仓库模型：`/home/lyx/train/leju/kuavo-ros-opensource/src/kuavo_assets/models/biped_s17/xml/biped_s17.xml`。
- 部署模型：`/home/lyx/train/leju/LejuLab-Deploy/src/leju_assets/models/biped_s17/xml/biped_s17.xml`。
- 部署配置示例：`/home/lyx/train/leju/LejuLab-Deploy/src/leju-controllers/leju-rl-controller/config/17/controllers/config_mimic_dance.yaml`。

数据集说明称约 39,685 条、206.4 小时、9.92 GB，分为 `qpos(T,30)`、`qpos(T,28)` 和 `data(T,30)` 三类；样本 NPZ 的字段和形状已抽查。按 50 Hz 计算，全库约 3,715 万帧。当前 `MotionLib` 会把全部轨迹和每帧跟踪 body 的位置/姿态/速度预计算到同一设备；仅假设 14 个 body 的四组 FK 输出就约需 37 GB 浮点内存，尚未计入原始轨迹、索引、回放缓存和训练网络，因此**不能直接把全库文件列表塞入现有配置**。

已用本机 MuJoCo 加载三份 XML：数据集模型为 `nq=30, nv=29, nu=23`，ROS 模型也是 `30/29/23`，部署模型为 `28/27/21`。将一条真实 30 列 NPZ 的首帧直接写入数据集模型后，`mj_forward` 成功且 body 状态有限。ROS 模型并非仅调整了顺序：与数据集模型相比，共同 body 的质量最大差值约 3.45 kg，左右膝下限也不同，抽样帧经名称重排后的末端位置有毫米级差异，不能默认动态等价。数据集 XML 自带 `floor`，但没有 `stand` keyframe；它可以独立用于 MuJoCo 回放，接入现有 UniLab 任务时需整理成机器人与场景资产，并定义 keyframe、命名碰撞 geom 和控制器。

数据集 `npz_format_description.md` 的 28 列小节有一句“去掉左右各 1 个手臂关节”，与同节列布局、后一句“去掉头部 2 关节”及数据集 viewer 的补头实现冲突。以实际列布局和样本检验为准：28 列保留腰、双腿和双臂，缺少两列头部；转换器应对此写明断言。

## 实施顺序与验收门槛

### 0. 固定模型与数据契约（先做）

1. 第一版选择 **数据集自带的 23 关节模型**，建立以其 MuJoCo 关节声明顺序为准的数据契约。30 列 `qpos`/`data` 可按原顺序使用；28 列 `qpos` 需在末尾补 2 个头部关节默认值。NPZ 中根四元数是 `wxyz`；接入本仓库运动库时，只在目标接口明确要求 `xyzw` 的位置转换。
2. 将 ROS 23 关节模型和 LejuLab-Deploy 21 关节模型列为后续对照，不在第一版混用。若要部署，按关节名把 23 关节轨迹/策略映射到 21 关节，并核对零位、轴向、限位、惯量及 base 坐标；模型动力学不同还需 sim-to-sim 验证，不能只删去头部两列。
3. 定义训练控制接口：策略输出是绝对目标、默认姿态偏置，还是参考轨迹残差；按关节名确定 `kp/kd`、`action_scale`、限幅，检查 MJCF 的 actuator 模式和仿真控制实现是否一致。硬件控制模式、方向和单位单独核对，不以 XML 的 `motor` 默认值代替部署约定。
4. 记录数据来源、许可和模型版本；训练集先选 `passed_motions.json` 中少量平地行走/站立/转向片段，不混入跳跃、爬行、外物交互或质检失败片段。

**门槛 A：** 23 关节数据集模型在目标 UniSim 后端能加载、站立姿态不自穿透；抽样轨迹回放时关节无系统性越限，脚底位置/根高度/朝向符合原轨迹。若模型/数据语义对不上，先停止移植算法。

### 1. 建立独立的 Roban2 任务与资产

1. 新增 `tasks/motion_tracking/roban2/supermimic/` 和对应 Hydra task 配置，保留 G1 任务作基线；注册独立任务名。场景指定 `base_link`、21 个关节与 actuator 名、目标 body、接触 geom、地面和 stand keyframe。
2. 复制或引用经过版本固定的数据集 MJCF/mesh，将现有 XML 中的地面与机器人配置整理为训练场景；补 `stand` keyframe，避免再添加第二个 `floor`。不要直接使用 `prepare_supermimic_assets.py`：它明确假设 G1 URDF、35 个 mesh、40 条 CSV 和 G1 的碰撞/关节数。此处不需要 URDF。
3. 将 `robot.py` 的关节顺序、默认位姿、跟踪 body、增益/缩放、碰撞名改成 Roban2 专属定义；用 MuJoCo 编译结果验证 `nq=30`、`nv=29`、`nu=23` 及每个 actuator 对应的关节。
4. 优先用本机 `mujoco` 后端做 1 环境 smoke test，再测 `mjwarp` 或 `motrix`；不能仅凭 XML 可被 MuJoCo 加载就认为其他后端支持全部几何和 actuator。

**门槛 B：** 环境创建、reset、单步动作及 scene 状态读取均成功；仿真中的关节顺序与任务映射一致，站立状态稳定。

### 2. 改造运动库和任务 MDP

1. 增加 NPZ 读取器和 schema 检查：识别 `qpos`/`data`、28/30 列、有限数值、四元数范数、帧率和片段长度；28 列在头部补默认值，形成 23 关节训练格式。保留每条轨迹的来源及筛选清单。
2. 从 120 fps 插值到 `ctrl_dt=0.02` 的 50 Hz；位置线性插值、根旋转 SLERP，随后在目标时间轴计算线速度、角速度和关节速度。短片段、异常跳变、重复帧和周期边界要被拒绝或显式处理。
3. 把 `G1Kinematics` 改成 Roban2 可用的 FK。当前实现假设每个相关 body 有一个关节，遇到固定父 body 会报错；必须与 MuJoCo `mj_forward` 做随机姿态的位置/朝向对照测试，再用于奖励和运动预处理。
4. 任务中的 `NUM_J=29`、`STATE_DIM=71`、`XI_DIM=33`、`OBS_DIM`、`GOAL_FRAME_DIM`、噪声/归一化向量，以及 `1:30`/`30:33` 等切片统一改为由任务规格推导。23 关节下状态长度应为 `13+2*23=59`；其余维度按实际观测定义计算。`manager_terms.py` 的 reset、动作、上下文随机化和 `env.py` 的默认 G1 碰撞集合也都要同步调整。
5. 跟踪 body、奖励权重、终止高度/倾角阈值、默认位姿和随机扰动尺度按 Roban2 重新标定；先让参考轨迹无动作回放的误差可解释，再训练策略。

**门槛 C：** 小轨迹集可 reset 到任意帧、播放、计算有限的 obs/goal/reward；FK 与 MuJoCo 一致；参考动作附近的奖励明显高于扰动状态，且不会无故提前终止。

### 3. 小规模训练，再扩容

1. 用数条已通过质检的短片段、少量环境、关闭编译运行训练 smoke test，验证收集数据、世界模型损失、策略梯度、checkpoint 保存/恢复和确定性评估。再逐步增加环境数和轨迹难度。
2. 将运动库改为按片段/类别分批加载、缓存或离线分片预处理，限定每次驻留的帧数和 FK 缓存；训练集与验证集按原始片段划分，避免同一片段的相邻帧泄漏。
3. 做行走/站立基线和跨片段评估，记录跟踪误差、跌倒率、关节限位/扭矩触发、吞吐、显存与内存。完成单类任务后再考虑大规模全身动作；物体交互还需要环境物体和接触建模，单靠轨迹文件不够。

**门槛 D：** 至少一个基础动作可稳定训练并在未见片段上评估；资源用量能随数据集规模受控增长。未通过则先定位模型/数据/控制契约，不能直接加大训练规模。

### 4. 如需真机部署，单独立项

现有 Leju 控制器配置使用 21 个关节，以及自己的观测项、关节方向、动作缩放、混合控制模式及 ONNX 推理。23 关节训练结果不能直接装入该控制器；应评估改为 21 关节重训、蒸馏，或实现经验证的动作映射。SuperMimic 当前策略还包含历史观测和未来目标，不能假定 checkpoint 可以直接由该控制器加载。需要逐字段制定训练到部署的观测、动作、时序和归一化映射，再做 ONNX 导出、sim-to-sim、限幅与安全停机验证；真机测试应最后进行。

## 是否继续使用本仓库

**建议继续的条件：** 目标是保留 SuperMimic 的世界模型与想象 rollout 方法；能投入任务级改造，并愿意先以几十条轨迹做闭环。项目的 `SuperMimicMdpProtocol`、网络和回放缓存支持按任务给出尺寸，值得复用。

**建议换仓库的条件：** 目标是尽快用 OpenLET 全量数据得到可接 Leju 现有 mimic 控制器的策略，或门槛 A/B 在短期无法通过。LejuLab-Deploy 的 `mimic_config_train_to_deploy.py` 明确面向 Isaac Lab 训练 `env.yaml`，说明现有部署链路已有配套训练约定；此时优先评估对应的 Leju/Isaac Lab 训练仓库。当前仓库没有现成的 Roban2 数据流和部署适配，全量数据的内存处理也需要额外设计。

**工作量粗估：** 门槛 A/B 的可行性验证约 1~2 个工程日；小规模训练闭环通常还需数个工程日；稳定支持多类别数据和部署映射可能需要数周。估计取决于模型语义是否一致、后端兼容性与实际算力，不能当作完成时间承诺。
