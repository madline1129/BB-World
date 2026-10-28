# BB-World

基于语义 Occupancy 的“大脑 + 小脑”场景预测研究代码。大脑从五帧历史 Occ 预测 actor 轨迹；小脑以当前 Occ 和运动条件为输入，直接生成未来六帧 Occ。

本仓库是当前研究实现的代码快照。数据集、模型权重、训练日志和机器连接配置不随代码发布；尚未提供开箱即用的数据下载器。机器专用的 detached launch 脚本未纳入，使用下文的标准 DDP 命令运行。

## 架构与参数量

| 模块 | 功能 | 参数量 |
|---|---|---:|
| 大脑 | Occ 编码、时序融合、256 个 actor query、分类与中心轨迹预测 | 2,299,171 |
| 小脑 | SE(3) 静态搬运、运动条件 FiLM、时间卷积、U-Net Occ 解码 | 1,145,516 |
| 可选 ego head | 从共享历史特征预测未来六帧完整 SE(3) | 87,204 |

基础系统共 **3.445M**；含可选 ego head 共 **3.532M**。默认小脑训练冻结大脑。

```text
历史 5 帧 GT Occ + 历史 ego 位姿
                 │
               大脑
       ┌─────────┴─────────┐
   actor 预测          可选 ego 预测
       │                   │
       └──────┬────────────┘
              │ + 当前 GT Occ
             小脑
              │
         未来 6 帧 Occ
```

默认实验使用未来 GT ego 位姿作为条件；预测 ego 模式使用新增预测头。未来 GT Occ 仅用于监督和评测，不作为模型推理输入。预测 ego 不会消除对历史 GT Occ 的依赖。

## 目录

- `occ_actor_brain_20261028/brain.py`：数据集、融合编码器、actor head、可选 ego head 和损失。
- `occ_actor_brain_20261028/prepare.py`：从 nuScenes 原始实例标注构造监督与完整 5+6 帧窗口。
- `occ_actor_brain_20261028/train.py`：DDP 大脑训练、验证、恢复及检查点。
- `occ_actor_brain_20261028/verify.py`：标签几何、掩码与输入隔离检查。
- `occ_actor_brain_20261028/check_ego.py`：ego 分支及小脑接口的 CPU 功能检查。
- `occ_cerebellum_20261028/cache_brain.py`：缓存冻结大脑的 actor/ego 预测。
- `occ_cerebellum_20261028/model.py`：小脑、Occ 损失和混淆矩阵指标。
- `occ_cerebellum_20261028/train.py`：DDP 小脑训练、验证与静态基线。
- `frozen_brain.py` / `brain_prepare_reference.py`：原始 actor 大脑与数据准备辅助函数的固定副本，用于原实验兼容。

两个源码目录应保持相邻，预测 ego 缓存脚本依赖该目录关系。

## 环境与数据

Python 3.10+，PyTorch CUDA 环境用于正式 DDP 训练。`requirements.txt` 是依赖下限列表，不是完整锁定环境。先按机器 CUDA 版本安装 PyTorch，再安装其余依赖。

```bash
pip install -r requirements.txt
python occ_actor_brain_20261028/check_ego.py
```

需要自行准备获准使用的 nuScenes 原始 `v1.0-trainval` JSON、Occ3D-nuScenes 标签、nuScenes 官方 split 定义，以及兼容的 `m0.bin`（训练）/`m1.bin`（验证）元数据缓存。仅对可信来源使用 pickle / torch.load。

元数据格式为 `{'infos': {scene_name: [frame, ...]}}`，每帧包含 `token`、微秒 `timestamp`、`ego2global_translation` 和 wxyz `ego2global_rotation`；Occ 路径为 `CACHE/s0/gts/SCENE/TOKEN/labels.npz`，`semantics` 为 `200×200×16` 整数标签，0–16 为语义类，17 为 free。这个缓存适配步骤目前需用户自行完成。

以下命令中的变量为用户指定的绝对路径：`META` 原始 JSON 目录、`CACHE` 元数据/Occ 缓存目录、`SPLITS` 官方 splits.py、`MANIFESTS` 输出清单目录、`BRAIN_RUN` 大脑运行目录、`BRAIN_CKPT` 大脑权重、`PRED_CACHE` 预测缓存、`SMALL_RUN` 小脑运行目录。

```bash
python occ_actor_brain_20261028/prepare.py \
  --metadata "$META" --cache "$CACHE" --splits-source "$SPLITS" --output "$MANIFESTS"
python occ_actor_brain_20261028/verify.py --manifests "$MANIFESTS"

torchrun --standalone --nproc_per_node=2 occ_actor_brain_20261028/train.py \
  --manifests "$MANIFESTS" --output "$BRAIN_RUN" --epochs 24 --batch-size 8

torchrun --standalone --nproc_per_node=2 occ_cerebellum_20261028/cache_brain.py \
  --brain "$BRAIN_CKPT" --manifests "$MANIFESTS" --metadata-cache "$CACHE" --output "$PRED_CACHE"

torchrun --standalone --nproc_per_node=2 occ_cerebellum_20261028/train.py \
  --cache "$PRED_CACHE" --output "$SMALL_RUN" --epochs 24 --batch-size 4
```

原 actor 缓存模式严格检查 checkpoint 为第 24 轮、31,704 steps，这是原实验的保护条件，不是通用任意 checkpoint 加载入口。正式训练前可在独立输出目录使用 `--smoke --max-steps 60` 做小样本检查。训练运行目录不要复用；只有明确续训时使用 `--resume`。

## 可选 ego 分支

先冻结 actor 分支，只训练新增 ego head。`PRED_CACHE` 中已有的未来 GT 位姿只作为 ego 标签，并按 token 校验顺序。

```bash
torchrun --standalone --nproc_per_node=2 occ_actor_brain_20261028/train.py \
  --manifests "$MANIFESTS" --ego-cache "$PRED_CACHE" --init "$BRAIN_CKPT" \
  --ego-only --output "$EGO_RUN" --epochs 24 --batch-size 8

torchrun --standalone --nproc_per_node=2 occ_cerebellum_20261028/cache_brain.py \
  --predict-ego --brain "$EGO_RUN/best_ego.pt" --manifests "$MANIFESTS" \
  --metadata-cache "$CACHE" --output "$EGO_CACHE"

torchrun --standalone --nproc_per_node=2 occ_cerebellum_20261028/train.py \
  --ego-source predicted --cache "$EGO_CACHE" --output "$EGO_SMALL_RUN" \
  --epochs 24 --batch-size 4
```

`EGO_RUN`、`EGO_CACHE`、`EGO_SMALL_RUN` 均须使用独立目录。预测模式缺少预测位姿时直接报错，不回退到未来 GT 位姿。该分支已通过功能检查，但在本快照发布时尚未正式训练，不能声称已有预测 ego 条件下的 Occ 成绩。

## 监督与评测边界

- 大脑使用额外 nuScenes actor 实例/中心监督，预测当前已存在的 actor，不显式预测未来新进入的对象；置信度是当前检测置信度，不是未来轨迹概率。
- 小脑损失为全网格加权 CE + 0.5 × 前 17 类 semantic Dice，无 camera/lidar visibility mask。
- 验证使用未来 1/2/3 秒的平均 occupied IoU 和 semantic mIoU；同时记录零类处理差异。actor mIoU 为八个类别的平均，包括停放车辆，不能等同于实际运动区域指标。
- 静态基线仅搬运当前 Occ；当前 GT ego 实验不能代表完全无未来位姿条件的性能。
- 未完成统一评测脚本的 II-World 配对复测，也未完成去掉大脑条件的贡献消融；不据此声明严格公平的 SOTA。

## 已完成验证

基础模型曾通过双卡小样本训练、参数同步及 checkpoint 恢复检查。新增 ego 分支通过 CPU 合成学习、SE(3) 合法性、actor 输出保持不变、预测位姿到小脑的前向通路，以及真实数据 token 顺序/标签隔离/冻结梯度检查。这些功能验证不等于 ego 分支已达到有效预测精度。
