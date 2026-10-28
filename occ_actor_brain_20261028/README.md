# Occ actor brain：第一阶段双卡基线

本轮只训练大脑，包含可训练的 Occ 编码、当前 actor 检测和未来六步中心预测。没有小脑、Occ 生成器、VAE、冻结 tokenizer，也不加载旧模型权重。

## 数据与接口

- 输入：5 张原始 GT Occ，形状 `[B,5,200,200,16]`，0.4m 体素；五帧历史自车位姿和历史时间戳。只访问过去和当前帧。
- 用完整 SE(3) 将历史 Occ 最近邻采样到第五帧 ego 坐标系；变换后的越界区域用独立 unknown 类，不能冒充 free。编码器仍然是大脑内部可训练的一部分。
- 标注：官方 nuScenes `sample_annotation.translation` 全局三维中心，按 `instance_token` 和目标 `sample_token` 精确连接，再变换到第五帧 ego 系；训练使用 xy，标注文件保留 xyz。
- 当前 actor 筛选：中心在当前 Occ XYZ 范围内且当前 lidar+radar 点数大于零，包含静止车辆。类别为 car、truck、bus、trailer、construction_vehicle、pedestrian、motorcycle、bicycle。
- 未来标注缺失仅影响监督 mask；不把缺失中心填零参与 loss，也不将标注缺失解释为 actor 消失。未来中心可以离开当前范围。
- 每个查询对应一条六步轨迹，查询容量 256；本数据当前 actor 最多 72 个，不做 GT 数量截断。
- `Brain.interface()` 输出 `[B,6,256,3]`，最后一维为 `(x,y,p)`。t 是六个关键帧索引，名义间隔 0.5s。p 当前定义为 `1-P(background)`，表示当前 actor 检测置信度，**不是已校准的未来轨迹概率**。用阈值筛选实际 actor 数量。
- 当前版本不预测未来新进入的 actor，不处理多模态轨迹。六个时间步共享当前 actor 检测置信度。

## 结构与训练

19 类 embedding（18 个原始类别 + unknown），高度维度保留后投影到 BEV；共享卷积空间编码器；五帧特征与时间编码融合；4 层 Transformer 查询解码；类别、当前中心和六步累计位移三个 head。约 230 万参数，全部一起训练。

训练时以当前类别和中心做 Hungarian matching，未来六步沿同一匹配身份监督。GT 类别、中心、实例 ID、actor 数量、未来位姿和未来 Occ 均不传入模型 forward。

损失：`2*class_CE + current_center_SmoothL1 + 0.5*future_center_SmoothL1 + 0.5*future_displacement_SmoothL1`；背景类别权重 0.1，缺失未来按 mask 排除。

双卡 DDP，BF16，AdamW，初始目标 LR 2e-4，200 step warmup 后 cosine 到 10%；梯度裁剪 5；每卡 batch 8，全局 16；24 epochs。每个 epoch 完整验证，step 20、每 500 step 和 epoch 末保存 `latest.pt`；另存检测召回率最好的 `best_recall.pt`。恢复功能保留模型/优化器/步数，但不承诺中途恢复的随机数逐位重放。

## 划分与评测边界

官方 700/150 train/val 场景集合已逐项比较；原始 sample 链的首帧、尾帧、帧数及逐帧 next 均检查。总帧数 28130/6019，只取完整 5+6 窗口，得到 21130/4519 个训练/验证窗口，不跨场景、不补重复边界帧，train/val sample token 不相交。当前阶段训练完整窗口的采样策略并不等同于 II-World 的循环序列训练。

Actor 验证同时报告 `confidence>=0.5`、同类别、中心距离不超过 2m 的 precision/recall，以及成功匹配 actor 上的 ADE/FDE 和静止中心基线。ADE/FDE 必须与检测召回一起看；匹配为空时为 null，不报告零误差。验证按 rank 划分不重复样本。

本轮使用额外实例/框监督，应明确披露。Actor 指标是辅助指标，**不能与 II-World 的 Occ IoU/mIoU 直接比较**；完整系统比较要等小脑接入后按其 1/2/3s Occ 评测协议执行。

## 文件

- `prepare.py`：原始标注提取、官方场景划分检查、manifest 与来源哈希。
- `brain.py`：数据加载、端到端模型、接口和监督。
- `verify.py`：坐标变换、类别完整性、身份唯一性、监督 mask 和 actor 排列不变性检查。
- `train.py`：双卡训练、小样本检查、完整验证和检查点。
- `launch.py`：独立后台启动，拒绝复用已有 run 目录。

官方元数据源：`https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval_meta.tgz`。
场景划分来源：`https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py`，下载副本及哈希已保存。

## 可选 ego 预测分支（2026-10-28）

`Brain(predict_ego=True)` 在共享时序 BEV 特征后添加 ego head。输入为全局池化的 BEV 特征、五帧历史相对 SE(3) 和历史时间，输出 `ego_future_to_current [B,6,4,4]`，方向为未来 ego 坐标到当前 ego 坐标。六个预测时刻固定为 0.5/1/1.5/2/2.5/3 秒。平移采用历史末两帧计算的恒速先验加学习残差；旋转向量经矩阵指数生成合法 SO(3)。输出包含 XYZ 平移和完整旋转，不把二维 actor 轨迹当成 ego 位姿。

未来 GT 位姿仅存在于监督 target 中。`OccActors(..., ego_cache=...)` 从现有小脑缓存取 GT 位姿标签，并严格校验全部 token 顺序。推理 `forward` 不接收未来 GT 位姿或未来 Occ。历史 GT Occ 输入仍保留，因此这不等于已经实现从相机输入的完整系统。

训练脚本支持 `--ego-cache CACHE --init ACTOR_CHECKPOINT --ego-only`：加载原大脑，冻结原有 actor 网络，仅训练 ego head。训练目标为平移 Smooth L1 加旋转矩阵平方误差；验证记录 XYZ ADE、3 秒 FDE 和旋转角误差，另存 `best_ego.pt`。不加 `--ego-only` 可联合训练，但默认推荐先冻结 actor，避免改变现有实验。所有新实验必须使用独立输出目录。

CPU 检查：`python check_ego.py`。覆盖恒速先验方向、SO(3) 合法性、平移及旋转监督可学习性、旧 actor 权重兼容与输出不变、checkpoint 重载、小脑预测位姿路径、未来 GT 位姿污染隔离和缺失预测时禁止 GT 回退。合成测试通过不代表真实数据上的预测精度。

当前状态：分支、训练入口和小脑对接已实现并检查；尚未进行 ego 分支正式训练，不存在可报告的预测位姿版 Occ 验证成绩。远端独立源码目录为 `/path/to/workspace/occ-ego-20261028/source/research/`，原有 actor/小脑运行目录未修改。
