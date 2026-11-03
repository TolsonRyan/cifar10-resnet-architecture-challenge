# CIFAR-10: Custom ResNet Under Five Million Parameters

在 Windows RTX 4060 Laptop 8GB 上从随机初始化完成的 CIFAR-10 课程竞赛项目。只使用本项目提供的官方训练数据；无外部训练数据、预训练权重、教师模型、集成或 TTA。

**最终独立验证准确率：96.64%（4,832/5,000），可训练参数：4,327,754。**

最终权重为 `runs/wide_mixup/best.pt`，预测文件为 `submissions/submission.csv`。Kaggle 尚未提交，因此没有排行榜成绩。

团队成员：**Sheng Tao（单人）**。Kaggle team name：**Sheng Tao**。GitHub 私有仓库：[TolsonRyan/cifar10-resnet-architecture-challenge](https://github.com/TolsonRyan/cifar10-resnet-architecture-challenge)；课程 Google Form 注册状态尚未核验。

## 实际实验结果

| 实验 | 参数量 | Epochs | 最佳 epoch | 训练期最佳验证 | 独立复评 | 训练时间 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline：32/64/128 通道，crop + flip | 1,084,586 | 60 | 59 | 94.04% | 94.04% | 249.77 s |
| Wide：64/128/256 通道，增加 Cutout(16) | 4,327,754 | 160 | 158 | 96.32% | 96.30% | 1,812.07 s |
| Wide + Cutout + Mixup(alpha=0.2) | 4,327,754 | 160 | 158 | 96.64% | 96.64% | 1,812.19 s |

时间来自逐轮计时，不包含安装、数据审计和单独复评/推理。另有 3 epoch 断点恢复试跑（13.77 s），不是正式候选。正式训练合计 3,874.03 s。

最终模型比 Baseline 提升 2.60 个百分点。两种 Wide 模型是仅改变 Mixup 的对照；最终模型在复评中比 Cutout 模型多正确分类 17 张图片（+0.34 个百分点）。这不是多随机种子上的统计显著性结论。Baseline 与 Wide 同时改变宽度、增强和训练时长，不能将全部提升归因于宽度。

Cutout 的训练期结果与单独复评相差 1 张图片。训练启用 BF16、cuDNN benchmark；单独推理的卷积算法可能不同，不保证跨环境逐位一致。两次独立复评最终模型均为 96.64%，两次最终预测 CSV 的 SHA-256 完全相同。

## 数据与合规

- 执行依据为目录中实际存在的 `ResNet_Codex_Brief_2026.md`；用户提到的 `Codex_ResNet_Project.md` 未在目录中找到。
- 全量审计 50,000 张训练图与 2,000 张测试图：均为 32×32 RGB，无缺失/多余文件，无重复 ID、文件名或像素完全相同的图片。
- `train_labels.csv` 为 `id,filename,label`，`test.csv` 为 `id,filename`，sample 为 `id,label`。10 类各 5,000 张。
- 固定 seed=2026 分层划分：45,000 train / 5,000 validation，每类分别 4,500 / 500 张；保存于 `results/split_seed2026.npz`。索引指向原始 CSV 行，原始 CSV 的 SHA-256 已记录。
- 划分 NPZ 保留在本地和交付包中，不上传 GitHub；远程仓库保留 seed、划分代码和预期哈希，执行 `src/audit_data.py` 可从官方本地数据重建。
- 归一化 mean/std 仅由 45,000 张训练图片计算；验证图片不参与梯度、归一化统计或增强训练；测试图片仅用于文件完整性检查及最终推理。
- 最终使用有真实独立验证结果的 45,000 张训练模型，没有使用全部 50,000 张重新训练。
- 最佳权重按最高验证 accuracy 选择，平局时按更低验证 loss。验证集用于 epoch/配置选择，不能将其当作完全独立的最终测试集。
- 每次构建与预测都断言 `sum(p.numel() for p in model.parameters() if p.requires_grad) <= 5_000_000`。
- 实时 Kaggle Rules 页面访问失败，合规依据为用户提供、已核对的说明。实际提交前还需用户确认账号/团队状态、当日剩余提交额度和是否有规则更新。

## 架构与训练

本地实现的 post-activation residual block：`ReLU(BN(Conv(ReLU(BN(Conv(x))))) + S(x))`。每个 stage 有 3 个 block，每个 block 有 2 个 3×3 卷积。stem 为 3×3/stride 1；stage 2、3 的首个 block 下采样，shortcut 使用 1×1 卷积与 BN；最后为 global average pooling 和 10 类线性层。并非直接调用标准 ImageNet ResNet，也不加载其权重。

最终配置：SGD + Nesterov，momentum 0.9，weight decay 0.0005，batch size 128，基础 LR 0.1，5 epoch warmup + 逐 step cosine，最低 LR 比例 0.001，160 epoch，BF16。增强为 reflect-pad 4 随机裁剪、水平翻转、16×16 Cutout（中心靠边时裁剪到图内）和 Mixup(alpha=0.2)，label smoothing=0。完整配置见 `configs/wide_mixup.json`。

Mixup 日志中的 `train_accuracy` 是相对两个混合目标的加权准确率，train loss 也是混合目标损失，不能与普通硬标签训练准确率直接对比。验证始终使用未增强图片与普通交叉熵。

每个 epoch 保存日志和 `last.pt`（模型、优化器、训练位置、调度配置及 Python/NumPy/Torch/CUDA 随机状态）；`best.pt` 保存最佳模型。逐轮记录时间、峰值已分配 GPU 显存、GPU 温度和功率。原始脚本快照保存在各 run 的 `source/`。

## Windows PowerShell 复现

从项目根目录执行。实测版本：Python 3.11.4、PyTorch 2.6.0+cu124、CUDA runtime 12.4、RTX 4060 Laptop。

### 使用本机已建立的环境

```powershell
.\.venv\Scripts\python.exe src/check_gpu.py
.\.venv\Scripts\python.exe src/predict.py --checkpoint runs/wide_mixup/best.pt
.\.venv\Scripts\python.exe src/predict.py --validate-only
.\.venv\Scripts\python.exe src/evaluate.py --checkpoint runs/wide_mixup/best.pt
```

本机 C 盘曾无法容纳 CUDA wheel 和解压文件，项目 `.venv` 是指向 `D:\CodexEnvs\cifar10-resnet-20261102` 的 junction。该专用环境使用 `--system-site-packages` 复用原有只读依赖；本项目覆盖的 CUDA torch 安装在专用环境中，原有 Python 安装未被修改。实际包版本保存在 `requirements.txt` 和 `results/requirements-resolved.txt`。

### 新机器安装

推荐创建完全独立的环境。确保环境与临时下载盘有足够空间；CUDA wheel 约 2.5 GB，安装时需要额外解压空间。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果 C 盘空间不足，可以将环境建在 D 盘并在项目根目录创建 `.venv` junction；不要覆盖一个已有环境。

```powershell
py -3.11 -m venv D:\CodexEnvs\cifar10-resnet-reproduce
New-Item -ItemType Junction -Path .venv -Target D:\CodexEnvs\cifar10-resnet-reproduce
New-Item -ItemType Directory -Force D:\CodexEnvs\cifar10-resnet-reproduce\tmp
$env:TEMP='D:\CodexEnvs\cifar10-resnet-reproduce\tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 数据、训练和断点恢复

本地数据应为 `data/train_images/`、`data/test_images/`、`data/train_labels.csv`、`data/test.csv`、`data/sample_submission.csv`，并保留根目录原始 `sample_submission.csv`。程序不会下载训练数据。

```powershell
.\.venv\Scripts\python.exe src/audit_data.py
.\.venv\Scripts\python.exe src/check_gpu.py
# 顺序运行，自动跳过完成的实验，从已有 last.pt 恢复未完成实验
.\.venv\Scripts\python.exe src/run_pipeline.py --configs baseline wide_cutout wide_mixup
# 单独从 checkpoint 恢复；配置必须与最初运行完全相同
.\.venv\Scripts\python.exe src/train.py --config configs/wide_mixup.json --resume
```

要从头重复同一配置，请复制配置并修改 `name` 为新实验名，避免覆盖原始证据。训练脚本遇到已存在日志而未指定 `--resume` 时会拒绝覆盖。更改 `epochs` 会改变 cosine 轨迹，不应冒充原实验恢复。

```powershell
.\.venv\Scripts\python.exe src/evaluate.py --checkpoint runs/wide_mixup/best.pt
.\.venv\Scripts\python.exe src/predict.py --checkpoint runs/wide_mixup/best.pt --output submissions/reproduction_check/submission.csv
.\.venv\Scripts\python.exe src/summarize.py
.\.venv\Scripts\python.exe src/finalize.py
```

`finalize.py` 是本次三组实验的交付审计脚本，会核验完成轮数、最佳 epoch、参数量、checkpoint、CSV 和两次预测是否一致；新增其他候选时应同步扩展其名单及最终选择逻辑。

## 文件与证据

```text
configs/                         完整配置
src/                             数据、模型、训练、评估、预测与审计
runs/<name>/                     原始日志、best/last.pt、代码快照、曲线
results/data_audit.json           全量数据检查与哈希
results/split_seed2026.npz        固定训练/验证划分
results/experiments/              可版本控制的日志、配置、评估与曲线
results/experiment_summary.csv   实验对照表（训练期与独立复评均保留）
results/final_model_metrics.json  最终指标、合规与预测检查
results/artifact_audit.json       交付物一致性审计
submissions/submission.csv        最终预测，未上传 Kaggle
submissions/submission.checks.json
report/aaai_report.tex            可修改的 AAAI 报告源文件
report/methods_notes.md           中文实验说明及课程交付待办
report/figures/                   曲线、对比与混淆矩阵
```

模型和数据不纳入 Git；需按老师允许的方式单独交付 checkpoint。GitHub 仓库已由用户创建，并核验为私有仓库；不向其他团队或公众开放。仓库包含代码、配置、逐 epoch 日志、评估、图表及报告源码。课程注册状态仍待核验。

本地完整交付包 `delivery/cifar10_resnet_artifacts.zip` 包含代码、报告素材、全部本次实验 checkpoint/日志和预测，排除图片数据集、Python 环境与 Git 历史。包内 `ARTIFACT_MANIFEST.json` 记录每个文件的 SHA-256；压缩包没有上传或公开。运行 `src/package_delivery.py` 可重新生成。

最终 checkpoint SHA-256：`844b4b53d9af832abc46db6043d488d41e8a490693765c1779f4efaba541d13b`。

最终 CSV SHA-256：`f3f48f7897c76cf2cbf334cbb37cc4aa22f1cd82ec73f5cad2832283ae1114ed`。

## 参考与工具使用

架构与方法参考 [ResNet](https://arxiv.org/abs/1512.03385)、[Wide Residual Networks](https://arxiv.org/abs/1605.07146)、[Cutout](https://arxiv.org/abs/1708.04552)、[Mixup](https://arxiv.org/abs/1710.09412)、[SGDR/cosine annealing](https://arxiv.org/abs/1608.03983)。本项目是本地实现，没有复制这些项目的训练权重或外部数据。依赖框架为 [PyTorch](https://pytorch.org/)，CUDA 安装参考其[官方版本说明](https://docs.pytorch.org/get-started/previous-versions/)。

代码和文档有 Codex 辅助，实际训练由本机 GPU 执行。工具使用说明见 `TOOL_USAGE.md`。学生需要理解、审阅和修改最终代码/报告，确认课程对辅助工具的要求；报告中的作者、队名和 GitHub 地址不可虚构。
