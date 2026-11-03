# 实验记录与课程报告素材

## 可以直接引用的结果

最终模型 Wide + Cutout + Mixup(alpha=0.2)，4,327,754 个可训练参数；在固定 5,000 张验证图片上正确分类 4,832 张，准确率 96.64%，独立复评交叉熵 0.13903644。最佳 checkpoint 来自 epoch 158，训练总计 160 epoch、1,812.19 秒。复评两次准确率一致，最终预测两次逐字节一致。

Baseline（1,084,586 参数、60 epoch）为 94.04%。Wide + Cutout（4,327,754 参数、160 epoch）的训练期最佳准确率为 96.32%，独立复评为 96.30%。差异为 1 张图片，应保留而非四舍五入掩盖。BF16 与不同 cuDNN 算法的数值差异意味着不保证逐位复现。

Wide + Mixup 相比 Baseline 多正确 130 张，+2.60 个百分点。相比相同 Wide/Cutout 配置，增加 Mixup 后复评多正确 17 张，+0.34 个百分点；81 张仅最终模型预测正确，64 张仅 Cutout 模型预测正确。该增益较小，未做多个 seed 的重复实验，不能宣称普遍显著提升。Mixup 模型的验证交叉熵高于 Cutout 模型，因此选择依据是比赛的 accuracy，而非最低 loss。

描述性 Wilson 95% 区间约为 96.10%–97.10%，未修正同一验证集上的多次模型/epoch 选择，不能当作隐藏测试准确率保证。

## 模型与方法

3×3 stem，3 个 residual stage，各 3 个含两个 3×3 卷积的 block。最终 stage 通道为 64、128、256；stage 2/3 首 block stride 2；维度变化用 1×1 卷积+BN shortcut，普通 block 为 identity shortcut。每个 block 为 post-activation 的 F(x)+S(x)。Global average pooling 后为 256→10 的线性分类头。

所有模型从随机初始化训练。仅使用官方训练图片建立固定 45k/5k 分层划分（seed 2026），每类验证 500 张。归一化仅来自训练部分：mean=[0.4915135944,0.4823147097,0.4468138380]，std=[0.2470509270,0.2434797189,0.2615095063]。

优化器 SGD/Nesterov，LR 0.1、momentum 0.9、weight decay 5e-4、batch 128，最终模型使用 5 epoch warmup + 160 epoch 逐步 cosine schedule，最低 LR 1e-4。反射 padding 4、32×32 random crop、随机水平翻转、Cutout 16、Mixup alpha 0.2。无 label smoothing。BF16 autocast，无梯度缩放；GPU 缓存 uint8 图片避免反复读取 52,000 个小文件。

最终使用单个 checkpoint 做标准推理；无 TTA、模型集成或外部教师。没有用全量 50k 再训练，从而保留可直接复评的最终权重。保存固定划分、所有配置、每 epoch 指标、源代码快照、best/last checkpoint 和随机状态。

## 实验设计边界

Baseline 到 Wide 同时改变了宽度、增强、warmup 和训练长度，属于配置对比，不是单变量架构消融。Wide Cutout 与 Wide Mixup 仅改变 Mixup alpha，构成受控增强对照。所有模型使用同一数据划分与 seed；不同增强会消耗不同随机数，不能说逐 batch 增强完全相同。

未运行多 seed 验证、交叉验证、另设独立开发集或对全部 50k 重训。隐藏测试图片没有参与训练、模型选择或统计量估计。无 Kaggle 成绩；本地验证分数不能冒充 public/private leaderboard。

最终模型每类准确率（类 ID 0–9）：97.4%、98.6%、95.2%、91.6%、97.4%、94.0%、98.4%、97.4%、98.8%、97.6%。类 3 与类 5 的双向混淆共 44 张。未对官方标签含义作额外外部映射。

## 工程问题与实际处理

- 两个既有 Python 环境均为 CPU-only PyTorch；建立了项目专用 CUDA 环境。
- 旧 pip 在 PyTorch index 上解析 typing-extensions 失败；只升级专用环境中的 pip 和相应依赖。
- C 盘安装 CUDA 时空间不足；删除本次未完成的专用环境，将新环境放至 D 盘并在项目内建立 .venv junction。原有数据、Python 和其他项目未被修改。
- 3 epoch pilot 先运行 1 epoch，保存退出，再恢复到第 3 epoch，验证恢复链路。正式训练按单 GPU 顺序执行。

这些是环境与工程记录，不应写成无关紧要的自动执行流水账占用报告篇幅。

## 图表与证据

- `figures/training_curves.png`：三组实验的验证 accuracy/loss。
- `figures/final_training_curves.png`：最终训练/验证曲线。Mixup train accuracy 为混合标签的加权指标。
- `figures/validation_comparison.png`：训练期最佳 accuracy 对比；精确独立复评数值见表格。
- `figures/confusion_matrix.png`：最终模型验证混淆矩阵。
- `../results/experiment_summary.csv`、`../results/experiments/`：真实配置与逐 epoch 日志。
- `../results/artifact_audit.json`、`../results/final_model_metrics.json`：证据一致性、参数及提交检查。

## 尚待用户完成的信息/外部操作

| 项目 | 实际状态 |
|---|---|
| 团队成员署名 | Sheng Tao，单人团队；用户已确认 |
| Kaggle team name | Sheng Tao；用户已确认，仍需与课程注册核对 |
| Google Form 团队注册 | 未核验；须由用户完成 |
| GitHub 远程仓库与访问权限 | https://github.com/TolsonRyan/cifar10-resnet-architecture-challenge；已核验为私有仓库 |
| Kaggle 正式提交 | 尚未提交；须用户明确批准 |
| 当前课程报告页数与模板年份要求 | 未取得更新版说明，待用户核对 |
| AAAI 报告 | 源文件及实验素材已准备；编译状态见 report/README.md |

不要虚构以上身份、注册、仓库链接或成绩。报告正文是可修改初稿，学生应自行理解、校订并补全；辅助工具说明独立保存在 `../TOOL_USAGE.md`。
