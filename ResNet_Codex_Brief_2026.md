# Deep Learning Project — CIFAR-10 ResNet Architecture Challenge
## Codex 本地项目执行说明（已核对 Overview / Data / Rules）

**比赛：** [NYU Tandon Deep Learning F26 — Efficient ResNet Architecture Challenge](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/overview)

**我的目标：** 在严格遵守比赛规则和 **5,000,000 个可训练参数上限**的前提下，用自己的 Windows 笔记本训练出尽可能准确、泛化能力强的 CIFAR-10 ResNet，并完成 Kaggle 预测文件、实验记录、可复现代码和课程项目报告素材。请优先保证模型真实可训练、提交格式正确、实验可复现，然后再追求排名。

## 0. 工作方式与边界

- 使用 **Codex 本地模式**，在我指定的 Windows 项目目录中开发，并优先使用本机 CUDA GPU。
- 电脑：**NVIDIA GeForce RTX 4060 Laptop GPU（8GB 独立显存），16GB RAM**。当前可用磁盘空间需要现场检查；不要把共享 GPU 内存算成额外 VRAM。
- 先检查 `nvidia-smi`、`torch.cuda.is_available()`、PyTorch/CUDA、剩余显存、RAM 和硬盘空间。尽量在项目内建立独立 Python 环境，避免破坏现有的 PyTorch 项目。
- 我会将比赛官方 dataset 一并放进项目目录（可能是 ZIP 或解压后的文件夹）。**首先检查并使用这些本地文件**；如为 ZIP，先验证并解压。不要因 Kaggle API 无法访问而忽略已提供的数据，也不要重复下载同一份数据。按实际 CSV 和图片检查数量、字段、ID 映射及样例提交格式。
- 只要没有明确的规则或技术阻碍，就主动推进：检查资料 → 准备数据 → baseline → 性能试跑 → 强化模型 → 验证/提交 → 交付。不要反复提出非必要的确认问题。
- **所有指标必须来自真实运行日志。** 不要编造 accuracy、排行榜成绩、训练时长或实验对比。不要将尚未完成的训练称为完成。
- 不要擅自访问或公开 Kaggle 凭据。若需要我接受比赛规则、登录或授权提交，请准确告知我需要操作的步骤。**实际向 Kaggle 提交预测前先向我确认**，除非我后续明确授权自动提交。
- 课程作业应当由我理解并独立完成/修改最终代码和文字；保留必要的工具使用及参考来源，不能把引用的实现冒充原创。

## 1. 比赛规则与团队要求（已核对）

**官方页面：** [Overview](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/overview) · [Data](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/data) · [Rules](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/rules)

### 1.1 核心限制（必须全部满足）
- 模型必须是包含 **residual / skip connections** 的自定义卷积神经网络（ResNet 或合理变体）。可修改残差块、通道宽度、卷积核、下采样、池化等设计。
- **最终模型所有可训练参数总数 ≤ 5,000,000**；包含分类头及其他可训练层。训练和导出预测前计算、记录并断言实际参数量。
- **从随机初始化训练**；不得加载任何外部预训练权重、预训练特征提取器或外部模型 checkpoint；禁止使用 ImageNet、CIFAR-100 或其他外部训练数据。
- 允许对合法训练数据做数据增强、优化器和调度器调整、正则化、模型结构优化、量化。**知识蒸馏**仅允许教师模型也使用合法数据从头训练的情况。
- 可参考公开代码和架构，但必须保留准确的来源引用；注意适用的软件许可与 Kaggle 的代码分享规则。
- 不能手工标注、从外部查询、推断或通过其他未授权方式获得隐藏测试标签；不能使用隐藏测试图片参与训练、验证或非法模型选择。
- 官方评价指标是 **classification accuracy**；公开榜和私有榜各对应隐藏测试集的 50%，**最终以 private leaderboard 排名**。不要仅为 public leaderboard 过拟合。

### 1.2 Kaggle Data 页面确认的真实文件与格式
- **50,000 张训练图片**，32×32 RGB、10 类；训练数据存放在 `train_images/`，标签文件为 `train_labels.csv`。
- **2,000 张无标签评估图片**，位于 `test_images/`，并提供 `test.csv` 中的图片标识信息。**不要把标准 CIFAR-10 的 10,000 张 test images 误当成本比赛的测试文件。**
- 官方提供 `sample_submission.csv`，预览中共有 2,000 条数据行，`id` 为 0–1999，`label` 是占位值 0。运行时仍以**实际下载的完整 CSV** 为准。
- Kaggle 预测文件必须命名为 **`submission.csv`**，并且**恰好两列**：`id,label`。`id` 必须与官方待预测图片一一对应，`label` 必须是 0–9 的整数；每张图片只有一条预测。严禁假定文件名排序天然等于 ID 顺序，需与 `test.csv` / sample 文件交叉核对。
- Data 页面列出的文件包括：`train_images/`、`test_images/`、`train_labels.csv`、`test.csv`、`sample_submission.csv`。允许的图片与标签应以这些官方文件为唯一依据。

### 1.3 团队与提交（Rules 页面确认）
- 每队**最多 3 人**；每个学生只能参加一个团队。
- **必须填写课程官方 Google Form 注册团队**，且注册的 **team name 与 Kaggle team name 一致**；团队成员变化需按课程规则获得批准。Codex 不能代替我完成这项身份/课程注册。
- **每队每天最多 5 次 Kaggle 提交**，额度由全队共用；正式提交前需获得我的明确同意，并记录提交使用的 config、checkpoint 和 CSV。
- 不得在不同团队之间私下分享训练权重、预测文件、竞赛代码或隐藏标签；遵守课程学术诚信及 Kaggle Foundational Rules。
- 页面已显示账号接受比赛规则；仍需在实际提交前确认本地 Kaggle 认证状态和团队状态，不能把页面状态当作 Codex 已获得账号授权。

### 1.4 GitHub、报告和合规复现（Rules 第 13–15 节）
- **必须提供 GitHub Repository**，包含模型、训练/评估/数据处理代码、增强方式、完整训练配置、最终参数量、复现预测的步骤及参考代码/方法引用。
- **必须提交课程项目报告**，覆盖：团队成员和 Kaggle team name、架构与设计依据、训练方法、不同配置比较、最终准确率及参数量、发现与局限、引用和 GitHub 链接。
- 课程老师可能抽查 **checkpoints、训练日志与实现细节**，优秀团队可能被要求复现实验；保存完整真实证据。
- **仓库分享注意：** Kaggle 基础规则对竞赛期间向其他团队私下分享代码、以及公开分享代码的地点有要求，而课程又要求 GitHub 仓库。先创建版本控制并准备 GitHub 仓库；在发布或向团队外人员开放仓库前，核对课程老师指定的访问方式和时间，避免违反分享规定。不要未经许可公开训练权重、预测文件或其他团队专有资料。
- 我此前提供的**课程 Grading Policy** 另要求报告采用 **AAAI 双栏格式**；这是课程格式要求，不是 Kaggle Rules 页明确指定的格式。请准备 AAAI 模板对应的可修改 LaTeX 报告，并核对课程是否有更新说明。

## 2. 数据使用与验证设计

- 只从比赛官方提供的 `train_images/`、`train_labels.csv` 加载有监督训练数据；从 `test_images/`、`test.csv` 加载真正的比赛测试图像。**不要额外下载 torchvision 中公开的 CIFAR-10 test split 参与训练或预测**，也不要混入其他数据集。
- 先验证 `train_labels.csv` 的列名、ID 到文件名的映射、标签范围与图片数量；不要凭经验假设 CSV 列名或图片编号。对缺失/重复图片与 ID 进行校验。
- 从 50,000 张允许的训练图片中创建固定、分层的训练/验证划分（例如 45,000 train + 5,000 val）；保存 indices、seed 与类分布，所有模型尽可能使用同一验证集。
- 训练时可用经规则允许的增强，例如 RandomCrop(32, padding=4)、RandomHorizontalFlip、Cutout/RandomErasing；验证/推理使用确定性的归一化和清楚记录的 transforms。不要把验证集或测试集引入训练过程。
- 核对 32×32 RGB 图像、0–9 类别编码、预处理是否一致。Kaggle 测试图片没有标签，不允许通过外部信息获取或推断。
- 预测时按照实际下载的 **`sample_submission.csv` 的 ID 和顺序**写出结果，确保 2,000 个 ID 与 `test.csv` / 图像文件严格匹配。不能仅按文件系统顺序生成预测。
- Public leaderboard 只能用于少量正式提交反馈，不能取代固定本地 validation set 进行系统调参。

## 3. 模型设计：先稳定，后优化

### 必须遵守的结构约束
- 设计**真正包含 residual/skip connections** 的模型，明确实现 `y = activation(S(x) + F(x))` 或合理的 pre-activation 残差变体。
- 只有**可训练参数总数 `sum(p.numel() for p in model.parameters() if p.requires_grad)` ≤ 5,000,000** 才能作为有效候选或提交模型；每次训练和生成提交文件前都用程序 assert。
- 所有权重从随机初始化开始训练（不是把参数全部初始化为零）；不得加载预训练参数。只能恢复本项目使用允许训练数据训练的检查点。
- 优先为 CIFAR-10 设计小图像友好的 3×3 stem、残差 stage、合理的下采样与 global average pooling；避免直接用超出参数上限的标准 ImageNet ResNet18。
- 请合理选择通道数、blocks 数和 downsampling，并**计算真实参数量**，不要根据架构名字猜参数数目。

### 建议的实验层次（由实际训练速度决定）
1. **Baseline：** 一个较小的自定义 ResNet（例如 CIFAR 风格的 3-stage 残差网络），确认训练、评估、预测链路全部正确。
2. **Strong candidate：** 在 5M 参数以内，尝试更合适的宽度/深度组合、优化器、weight decay、cosine learning-rate schedule、warmup、合理增强。
3. **少量受控消融：** 优先比较最有可能产生收益的 2–4 项因素（如结构宽度与深度、增强、训练 schedule），保持其他变量尽可能一致。不要先盲目跑数十组大型训练。
4. 选取真实 validation 最优候选，再根据剩余时间决定是否用全部合法训练集从头训练最终版本；如这样做，需保留先前验证方案的记录，不得再用 hidden leaderboard 代替本地验证。

不要盲目承诺 95%+ accuracy 或榜单名次；以可复现、真实的模型与泛化性能为准。

## 4. 本地 GPU 训练工程

- PyTorch + CUDA；优先利用支持的 **BF16 autocast**（若不稳定再尝试 FP16+GradScaler），使用适当 batch size（如 128，实际由显存决定）。
- 先做 **1–3 epoch smoke/pilot**：验证 forward/backward、loss 下降、无 NaN/OOM、计算性能和峰值显存；真实估算完整训练时间。
- 根据吞吐量、机器温度和可用训练资源决定正式训练预算，例如从 50–100 epochs 的 baseline 出发，在计算资源允许的情况下对更有潜力的候选训练 150–250 epochs；这些是**建议范围，不是比赛硬性规定**。
- 提供明确的 `--config` / CLI 参数，`seed`、optimizer、momentum/betas、LR、weight decay、batch size、epochs、scheduler、augmentation 均写入每次实验的配置。
- 每次实验保存 `last` 和按**最低 validation loss 或最高 validation accuracy**确定的 `best` checkpoint，记录训练/验证 loss 和 accuracy、每 epoch 耗时、峰值 GPU 内存与总可训练参数数目。
- 支持从 checkpoint 恢复（optimizer、scheduler、epoch 和随机状态尽可能保存），避免中断后重新训练已完成 epoch。
- 监控 GPU 显存与温度；避免两个大型训练同时争抢 8GB VRAM。合理处理 Windows DataLoader worker 设置、日志编码和路径分隔符。
- 显存不足时依次考虑减小 batch、混合精度、梯度累积等措施；**不可为省显存而偷偷突破或规避模型限制**。

## 5. 验证、最终推理与 Kaggle 提交

- 输出每个实验的训练/验证曲线，并整理实验对照表：**模型、可训练参数数目、数据增强、优化器、训练轮数、最佳验证 accuracy、训练耗时**。
- 最终模型主要根据固定 local validation 表现、泛化趋势及稳定性选择，而不是依赖 public leaderboard 的小幅波动。
- 加载正确的训练 checkpoint，使用 `model.eval()` 和 `torch.inference_mode()`（或 `torch.no_grad()`）生成预测。
- 按官方 `sample_submission.csv` **完整保留原始 `id` 与行顺序，仅替换 `label` 列**。输出文件必须叫 **`submission.csv`**，恰好为 `id,label` 两列；验证 **2,000 行**、无空值、无重复 ID、全部标签为整数 0–9、ID 与 `test.csv` 一致。
- 推理时如尝试 test-time augmentation、ensemble 或其他策略，必须先核对具体规则是否允许；不确定时优先单个合规模型的标准推理。不要对隐藏测试图像进行手动标注、训练、微调、非法选择或泄漏。
- **Kaggle 每队每天最多 5 次提交**。先本地校验 CSV，再安排少量有意义的提交；真实提交前须我确认或后续明确授权。仅实际获得的 Kaggle 回执分数才能作为 public leaderboard 成绩。
- 保存每次提交的文件、对应 checkpoint/config、提交时间及 Kaggle 回执。Private leaderboard 结果未发布前不得声称已取得私榜成绩。

## 6. 可复现代码、GitHub 与报告交付物

请建立简洁的课程项目目录（可按实际需要调整）：

```text
cifar10-resnet-project/
  README.md
  requirements.txt
  configs/
  src/
    model.py
    data.py
    train.py
    evaluate.py
    predict.py
  runs/
    baseline/
    candidate_*/
  results/
    experiment_summary.csv
    training_curves.png
    validation_comparison.png
    final_model_metrics.json
  submissions/
    submission.csv
  report/
    methods_notes.md
    figures/
    aaai_report.tex
  .gitignore
```

**交付要求：**
1. **完整、可运行的 Python 项目**及真实依赖清单；README 提供 Windows PowerShell 从环境、数据准备、训练、评估到生成官方 `submission.csv` 的确切复现命令。
2. **真实训练完成的合法 ResNet checkpoint**（≤ 5M trainable params）、最终 config、训练/验证日志及必要中间结果。合理设置 `.gitignore`，避免误提交数据集、密钥和超大权重；按老师的实际要求保存/交付 checkpoint。
3. 简洁的实验对比 CSV、训练/验证曲线、最佳模型参数量、合法本地验证精度及真实计时。
4. 与比赛官方 `sample_submission.csv` 一致的 **2,000 行 `submission.csv`**，以及自动验证脚本；在我授权后再上传 Kaggle。
5. **GitHub Repository**：包含模型与数据处理代码、超参数、参数统计、复现实验的说明、适当引用；在对外分享/设定权限前按课程和 Kaggle 分享规则核对。
6. **AAAI 双栏项目报告初稿或素材**：包括团队成员、Kaggle team name、设计依据、具体架构、数据划分、训练配置、实验比较、最终 Accuracy/参数量、失败尝试与局限、GitHub 链接和引用。仅使用真实数据，文字应自然、具体、避免自动化过程说明。
7. 列出团队 Google Form 注册、Kaggle 队名一致性、课程报告与 GitHub 的实际交付状态；**不要替我填写注册信息或声称已提交**。

## 7. 执行优先级与进度反馈

**第一阶段：** 检查比赛真实 Data 文件、CSV schema 和 Rules；验证本地 CUDA/依赖；加载官方 50,000 张训练图像并划分 validation；实现合法的 <5M ResNet，跑通 smoke test 和 2,000 张评估图片的预测链路。

**第二阶段：** 完成 baseline 正式训练，得到真实 local validation accuracy、曲线、参数量、模型 checkpoint 和可提交的 **`submission.csv`**。这是首要里程碑：即使高级优化未完成，也要留有一个合规、可复现的候选。

**第三阶段：** 根据 GPU 预算做少量高价值结构/增强/优化实验，选择最佳泛化候选；如获我授权，按每队每天 5 次的上限安排正式 Kaggle 提交。

**第四阶段：** 整理 GitHub 仓库、可复现实验记录与 **AAAI 双栏报告**；确认 Google Form 团队注册、Kaggle 名称一致性及课程要求的 GitHub/报告交付。公开/共享仓库前核对 Kaggle 的代码分享规定与课程交付安排。

每个里程碑简要汇报：已完成任务、真实 accuracy 和耗时、模型参数数目、当前问题、下一步。不要生成冗长的“自动执行日志”冒充课程报告，也不必在无关紧要的小决定上反复确认；但涉及账号授权、团队注册和正式 Kaggle 提交须经我决定。

**完成标准：** 合规的 ≤5M 参数且从头训练的 ResNet、真实可复现的本地验证结果、匹配官方 **2,000 张测试图片**的 `submission.csv`、GitHub 仓库、报告素材及完整来源引用；实际 Kaggle 成绩只在提交后引用回执。尚未完成或待我操作的项目要明确列出。
