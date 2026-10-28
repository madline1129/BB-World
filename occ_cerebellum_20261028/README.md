# 小脑第一阶段：固定大脑，直接预测未来 Occ

当前实现沿用已讨论的接口：大脑读取五帧历史 GT Occ；小脑只读取第五帧当前 GT Occ，以及大脑预测的运动条件。未来 ego pose 按已讨论的 II-World 条件设定提供。没有 VAE、离散 tokenizer、预训练 latent 或 latent diffusion。

## 固定的大脑

采用大脑第 24 轮 `latest.pt`（31704 steps），复制为独立不可混淆的 `weights/brain_epoch24.pt`，本轮不修改该权重。SHA256：

`ba62ec25028f06548a8bb7926ec480da7232ee3286180f1dcf801f95d5a05a56`

使用训练/验证各自的真实五帧输入，eval 模式冻结推理。缓存内容为：预测当前中心 `[256,2]`、预测未来中心 `[256,6,2]`、当前检测置信度 `[256]`。缓存没有用 GT actor 中心、身份或轨迹替代预测，也不传递大脑隐藏特征。

缓存预测与在线冻结大脑是同一阶段计算；缓存用于避免每个 epoch 重复读取五帧历史。小脑本身只读一张当前 Occ。固定采用当前检测置信度 >=0.5 的 actor；当前中心作为运动对应的起点，是大脑输出，不是新增 GT 输入。

## 小脑结构（1,145,516 个参数）

1. **静态几何搬运。** 给定未来六帧 ego pose，以完整 SE(3) 将当前 Occ 最近邻采样到各未来 ego 网格。越界标为独立 unknown，不能冒充 free。这是刚体几何先验，不使用未来 Occ 内容。
2. **运动条件。** 把大脑输出的当前/未来 xy 中心变换到各未来 ego 平面，按预测置信度加权、高斯半径 sigma=1.2m，得到 50×50 四通道条件：当前中心密度、目标中心密度、与当前中心关联的 dx、dy。位移以 10m 归一化。脑接口只有 xy，因此这部分使用 z=0 的平面投影；Occ 搬运仍使用完整三维变换。
3. **Occ 编码。** 19 类 ×4维 embedding，高度16层折叠成64通道；共享二维 U-Net 编码器，分辨率 200→100→50、通道32→64→128。每级带 GroupNorm、SiLU 和卷积残差块。
4. **条件调制。** 运动条件经两层 CNN 编成32通道，分别在瓶颈和两级解码特征做空间 FiLM。未来 ego 变换及预测时间另经 MLP 做瓶颈调制。瓶颈含沿六个时间步的 `3×1×1` 卷积，联合处理未来六帧。
5. **直接 Occ 解码。** 跳连 U-Net 解码回 200×200；输出288通道并展开为18类×16层高度，得到 `[B,6,18,200,200,16]` logits。静态搬运类别作为强度2的固定 logit prior，网络学习其上的语义修正。未知区域没有该先验。没有预训练编码器或独立 latent 生成阶段。

当前版本通过空间条件学习运动，没有按 actor GT box 做刚体搬运，也没有使用未来 GT flow。固定大脑本身暂不预测新进入的 actor；小脑可以根据当前上下文生成变化，但其显式运动条件缺少这些新对象，属于已知限制。

## 监督与评测

- 所有六帧未来 GT Occ 只作为监督目标，保持各自未来 ego 坐标系。
- 损失：18类全网格加权交叉熵 + 0.5×前17类 semantic Dice loss；Dice 对当前批次有 GT 的语义类求平均，不包含 free。交叉熵类别权重记录在 `model.py`，沿用已核对的 II-World 配置数组。
- 不使用 camera/lidar visibility mask；未来 GT actor 轨迹、框、身份均不进入小脑训练输入。
- 训练21130完整窗口，验证4519完整窗口，沿用已核对的700/150场景集合；没有跨场景、重复边界帧或验证 DistributedSampler padding。
- 每轮完整验证1/2/3秒的 occupied IoU 和 semantic mIoU，三时刻平均。分别记录 II-World 原生“排除恰好为零的类 IoU”口径和保留零分的修正口径；另报8类 actor 的类均值 IoU。
- 第一轮额外评测“当前 Occ 仅按未来 ego 位姿做静态变换”的基线，越界 unknown 在该基线输出时按 free 处理。
- 本实验用了额外实例监督训练大脑，完整窗口训练采样也不同于 II-World 循环序列训练，不能声明完全相同的监督/训练协议。可比较的 Occ 指标需同时披露这些条件。

## 正式训练

双卡 DDP / BF16 / AdamW，从零初始化小脑（不继承 smoke 权重）；大脑保持固定。24 epochs，每卡 batch4、全局batch8，LR 2e-4，warmup200步后 cosine 到10%，weight decay0.01、梯度裁剪5。每轮验证，按修正 mIoU 保存 `best_miou.pt`；step20、每1000step、epoch末保存含优化器的 `latest.pt`。续训使用 `--resume latest.pt`；不承诺中断后随机数逐位复现。

远端源码：`/path/to/workspace/occ-cerebellum-20261028/source`

远端数据/产物：`/path/to/experiments/occ-cerebellum-20261028/`

- `cache/`：冻结大脑的预测条件与来源哈希。
- `weights/brain_epoch24.pt`：固定大脑权重。
- `smoke-v1/`：小样本检查与检查点恢复证据。
- `run-v1/`：正式日志、配置、每轮验证结果、小脑权重。

## 已完成检查

双卡小样本60步，固定批次 loss 2.00497→0.96975；Occ embedding、条件编码和输出解码均有非零梯度。关闭运动条件后 logits 平均绝对变化0.00543；这是条件连通性检查，不代表条件提升了验证精度。两卡参数完整SHA256一致，重载检查点产生相同的语义预测，优化器状态可加载。

另通过：恒等及0.4m位移的三维采样、unknown边界、未来坐标系中的actor位移方向、零置信度条件清空、独立混淆矩阵IoU算例。小样本验证只用于代码检查；其结果不作为正式模型精度。

## 可选预测 ego 条件

新增模式不改变原来默认的 GT 位姿实验：

1. 先在 `occ_actor_brain_20261028/train.py` 使用 `--ego-cache`、`--init` 和 `--ego-only` 训练大脑的 ego 分支，得到 `best_ego.pt`。
2. `cache_brain.py --predict-ego --brain best_ego.pt ...` 在全新的缓存目录生成 actor 与 ego 预测。GT 位姿保留在 `future_to_current` 供标签/对照使用，预测位姿单独存于 `predicted_future_to_current`。两个源码文件夹须保持相邻结构。旧缓存不能覆盖。
3. 小脑 `train.py --ego-source predicted --cache NEW_CACHE --output NEW_RUN ...` 只将预测位姿作为输入；默认 `--ego-source gt` 保持旧行为。预测字段缺失时报错，禁止静默回退到 GT。续训禁止跨 ego 模式切换。

未来 GT Occ 仍用于训练监督和测评，从不作为推理输入。预测 ego 实验需重新报告 Occ 指标，不能继承 GT ego 实验分数。actor 和 ego 的第一版输出都是单一预测，没有未来驾驶意图或多模态轨迹分布。
