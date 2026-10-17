# PADC 核心文献复现任务书

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: `plan`
- Origin Date: `2026-07-23`
- Verification Status: `SOURCE-AUDITED; EXPERIMENTS UNVERIFIED`
- Version Label: `code_plan_v1`

## 1. 唯一核心复现对象

**Wenyu Zhang et al., “Predictive and Adaptive Deep Coding for Wireless Image Transmission in Semantic Communication,” IEEE Transactions on Wireless Communications, 2023.**

- DOI: [10.1109/TWC.2023.3234408](https://doi.org/10.1109/TWC.2023.3234408)
- 官方代码：[wyzhang-ustb/Predictive-and-Adaptive-Deep-Coding-for-Wireless-Image-Transmission-in-Semantic-Communication](https://github.com/wyzhang-ustb/Predictive-and-Adaptive-Deep-Coding-for-Wireless-Image-Transmission-in-Semantic-Communication)
- 本地论文：
  `D:/Paper/语义入门资料/code+paper（写代码重点看）/Predictive and Adaptive Deep Coding for Wireless Image Transmission in Semantic Communication/Predictive_and_Adaptive_Deep_Coding_for_Wirele.pdf`
- 本地代码包：
  `D:/Paper/语义入门资料/code+paper（写代码重点看）/Predictive and Adaptive Deep Coding for Wireless Image Transmission in Semantic Communication/0Predictive-and-Adaptive-Deep-Coding-for-Wireless-Image-Transmission-in-Semantic-Communication-main.zip`
- PDF SHA256：
  `2801E701D73E2D4D7E143378F87779BB8E5257CACF8C51A00CA7C5B51AC1358B`
- ZIP SHA256：
  `7685FEF43A53EE903E9B95F494BEE9DCA7C23E1E791710129F2E97594DDA6325`

## 2. 为什么先复现 PADC

PADC 是目前最适合成为新方向代码地基的论文：

1. 它包含完整的问题链：
   `DeepJSCC-V 可变码率编码 → OraNet 性能预测 → 最小 CR 选择`。
2. 它是组内 2025 年 DF-JSCC/PreNet 工作的直接机制前身。
3. 后续创新可以明确地把：
   - `PSNR 点预测`替换为`层级任务语义风险预测`；
   - `PSNR 约束下最小 CR`替换为`目标语义风险下最小 CR`。
4. 本地包包含训练脚本、测试脚本、CIFAR100 文件和部分 checkpoint，具备实际启动条件。
5. Dynamic-JSCC 更适合作为 PADC 的比较基线；WITT 更适合作为后续强编码器基线，而不是第一篇唯一核心复现对象。

## 3. 当前材料审计结果

### 3.1 本地代码包已有内容

- `DeepJSCC_V_train_CIFAR10.py`
- `DeepJSCC_V_test.py`
- `DeepJSCC_V_train_ImageNet.py`
- `Oracle_train_CIFAR10.py`
- `Oracle_train_ImageNet.py`
- `Oracle_test.py`
- `OracleNet_test_Kodak.py`
- `OracleNet_test_Kodak_image_level.py`
- `ADJSCC_train.py`
- `ADJSCC_test.py`
- `models.py`
- `OracleNet.py`
- `GDN.py`
- CIFAR100 Python 数据文件
- 5 个 DeepJSCC/DeepJSCC-V checkpoint
- 2 个 OraNet checkpoint

### 3.2 已有 checkpoint

```text
DeepJSCC_5x5_AWGN_3_256.pth.tar
DeepJSCC_VLC_5x5_AWGN_256_20.pth.tar
DeepJSCC_VLC_5x5_Fading_256_20.pth.tar
DeepJSCC_VLC_5x5_Fading_256_20_ImageNet.pth.tar
OracleNet_Fading_ImageNet.pth.tar
OracleNet_Fading_Res.pth.tar
```

### 3.3 当前不是开箱即用

已确认的问题包括：

- 多个脚本写死原作者 Windows 绝对路径；
- 测试脚本的 `CHANNEL` 与所加载 checkpoint 不一致；
- 某些脚本期望的 checkpoint 文件名在代码包中不存在；
- `utils.py` 通过 Keras 下载 CIFAR10/CIFAR100，而不是读取包内 CIFAR100；
- 没有 `requirements.txt` 或 Conda 环境文件；
- 没有 Kodak24 图像；
- 没有 ImageNet 数据；
- 没有 AWGN ImageNet 版 DeepJSCC-V/OraNet checkpoint；
- 没有论文 Figure 8 所需的 BPG+LDPC 实现；
- 若干结果累加/平均代码需要核查，例如以最后一个 batch index 作为除数；
- 原脚本没有统一 CLI、随机种子、配置快照和逐样本结果输出。

因此，现有材料支持**核心机制复现**，但尚不足以直接声称“整篇论文所有图完全复现”。

## 4. 论文原始实验设置

根据论文原文：

- DeepJSCC-V 训练集：CIFAR10 train；
- 验证集：CIFAR10 test；
- 测试集：CIFAR100 test；
- batch size：128；
- optimizer：Adam；
- learning rate：`1e-4`；
- DeepJSCC-V：最多 400 epochs；
- OraNet：最多 150 epochs；
- 训练 SNR：`U(0, 27) dB`；
- 训练 CR：`U(0.05, 0.5)`；
- 信道：AWGN 与 slow Rayleigh fading；
- CIFAR 测试的关键 CR：
  `{1/16, 1/14, 1/12, 1/10, 1/8, 1/6, 1/4, 1/3}`；
- ADJSCC 对照 CR：
  `{1/16, 1/12, 1/6, 1/3}`；
- 高分辨率训练：
  从 ImageNet validation 随机选 45,000 张训练、5,000 张验证，中心裁剪为 `128 × 128`，batch size 32，50 epochs；
- Kodak24：24 张 `768 × 512` 图像，每个传输点重复 100 次；
- Kodak Figure 11 的 CR：
  `{1/16, 1/12, 1/8, 1/6, 1/4}`；
- 实例级 CR 搜索：
  `{1/3, 1/4, ..., 1/20}`。

## 5. 我们要完成什么

复现分为“核心复现”和“完整扩展复现”。先完成核心复现，验收后再决定是否补齐论文所有外围对照。

### R0：建立可追溯复现副本

- 将原 ZIP 原样保存为只读来源；
- 在当前项目中建立独立目录：
  `reproduction/padc/`；
- 记录原始文件哈希；
- 外部代码明确标为 baseline reference，不并入当前 `STATE_DIM=452` 主线；
- 建立独立环境，不污染当前 `pytorch` 环境。

产物：

```text
reproduction/padc/ORIGINAL_SOURCE_MANIFEST.json
reproduction/padc/ENVIRONMENT.md
reproduction/padc/KNOWN_ISSUES.md
```

### R1：预训练 checkpoint 冒烟复现

- 修复路径和配置问题，但不改变算法；
- 验证 AWGN/Fading DeepJSCC-V checkpoint 可加载；
- 验证 Fading OraNet checkpoint 可加载；
- 在 CIFAR100 上输出 PSNR；
- 在 Kodak24 上输出真实/预测 PSNR；
- 导出逐样本、逐 SNR、逐 CR 的 CSV。

验收：

- 全部 checkpoint 可重复加载；
- 固定 seed 后输出稳定；
- 不出现 NaN/Inf；
- 图像输入、输出尺寸正确；
- 结果文件、配置和日志完整。

### R2：从头训练 DeepJSCC-V

分别训练：

- CIFAR10 + AWGN；
- CIFAR10 + slow Rayleigh fading。

要求：

- 严格按论文 `400 epochs / batch 128 / Adam / lr 1e-4`；
- 保存 best checkpoint，而不是只保存最后 epoch；
- 至少记录 train/validation MSE、PSNR、CR 和 SNR；
- 训练中断可恢复；
- 所有随机源可控。

### R3：从头训练 OraNet

分别基于 R2 的两个 DeepJSCC-V 训练：

- AWGN OraNet；
- Fading OraNet。

要求：

- DeepJSCC-V 冻结；
- 输入保持论文定义：
  channel-wise mean、channel-wise std、SNR、CR；
- 目标为逐图像真实 PSNR；
- 最多 150 epochs；
- 输出 MSE、MAE、校准散点和误差分布。

### R4：复现 CIFAR 核心结果

至少复现：

- Figure 4：DeepJSCC-V 与 ADJSCC 的 PSNR-SNR 曲线；
- Figure 5：给定目标 PSNR 时的数据级最小 CR；
- Figure 7：OraNet 在 CIFAR100 上的 PSNR 预测 MSE。

其中 Figure 4 需要训练 4 个 AWGN ADJSCC 和 4 个 Fading ADJSCC 模型；如果先做最小核心闭环，可先选 `{1/16, 1/6, 1/3}` 三个 CR，随后补齐 `1/12`。

### R5：复现 Kodak 核心结果

至少复现：

- Figure 11：5 个 CR 下预测 PSNR 与实测 PSNR；
- Figure 13：实例级 CR optimizer，至少包含 `kodim13` 和 `kodim23`；
- Fading 可先使用本地 ImageNet checkpoint；
- AWGN 需要补充 AWGN ImageNet checkpoint，或提供 ImageNet 数据后从头训练。

### R6：完成正式复现报告

最终报告必须包含：

- 论文方法逐模块对应关系；
- 原代码缺陷与修复清单；
- 每处修改是否改变算法；
- 环境、GPU、依赖和随机种子；
- 数据来源与哈希；
- 训练曲线；
- 论文曲线与复现曲线叠加图；
- 逐图/逐 SNR/逐 CR 数据；
- checkpoint 哈希；
- 成功、部分成功和未复现项；
- 一键运行命令。

最终产物建议：

```text
reproduction/padc/
  original/
  patched/
  configs/
  checkpoints/
  results/
  figures/
  scripts/
  ORIGINAL_SOURCE_MANIFEST.json
  REPRODUCTION_MANIFEST.json
  FULL_REPRODUCTION_AUDIT.md
  README_REPRODUCTION.md
```

## 6. 什么算复现成功

### 核心成功标准

1. **DeepJSCC-V 主趋势**
   - PSNR 随 SNR、CR 单调改善；
   - 与论文 Figure 4 的方法排序和曲线趋势一致；
   - 可读 operating points 的 PSNR 与论文图差异优先控制在 `±0.5 dB`，从头训练放宽到 `±1.0 dB`，超出必须解释。

2. **OraNet**
   - CIFAR100 在 `SNR > 5 dB` 的主要区域预测 MSE 接近或低于 1；
   - Kodak24 各 CR 的 MAE 目标低于 `0.5 dB`；
   - 不只报告平均误差，还报告最差图像和误差分布。

3. **CR optimizer**
   - 预测最小 CR 与穷举 oracle 最小 CR 相同或相邻一个 CR 档位；
   - 目标 PSNR 约束违反率不超过 5%，并单独报告低 SNR 条件。

4. **可重复性**
   - checkpoint 测试重复运行差异应来自明确的信道随机性；
   - 使用固定成对信道样本后结果可复现；
   - 从头训练至少完成 1 个完整 seed，并对关键 operating points 补 3 seeds。

5. **可审计性**
   - 每个数字能追溯到配置、checkpoint、数据划分和日志；
   - 不能用论文数字替代本地未跑出的结果。

## 7. 我需要你提供或确认什么

### 7.1 必须确认

1. **允许我在当前项目中建立独立目录**

   ```text
   D:/PyChram_project/CognitiveSemCom/reproduction/padc/
   ```

2. **允许建立独立 Conda 环境并安装依赖**

   建议环境名：`padc_repro`。  
   当前 `pytorch` 环境是 Python 3.13，且缺少 `scikit-image`；不建议污染它。

3. **允许下载公开小型数据**

   - CIFAR10；
   - CIFAR100（本地已有一份，但仍需统一读取方式）；
   - Kodak24。

4. **GPU 可用窗口**

   当前 GPU 为 RTX 5060 Ti 16GB，容量足够。需要允许连续或分阶段占用 GPU；实际时长先通过 1 epoch profiling 再估算，不预先虚构训练时间。

### 7.2 如果组内有，请优先提供

- 博士或组内已经跑通 PADC 的工作目录；
- 对应的 Conda `environment.yml` 或 `pip freeze`；
- 实际运行命令；
- Kodak24 文件夹；
- AWGN ImageNet DeepJSCC-V checkpoint；
- AWGN ImageNet OraNet checkpoint；
- 组内复现曲线、日志或结果表；
- 对原代码做过的修改说明。

这些材料不是 CIFAR 核心复现的必要条件，但会显著降低重复排错成本。

### 7.3 “整篇论文全部图”额外需要

- ImageNet validation 数据的合法本地访问路径；
- BPG 编码器；
- 802.11n LDPC 与 BPSK/4QAM/16QAM 对照实现；
- 足够时间训练 8 个 ADJSCC、2 个 DeepJSCC-V、2 个 OraNet，并完成 ImageNet fine-tuning。

如果没有这些，Figure 8 的 BPG+LDPC 和 AWGN Kodak 全链路不能称为原论文完整复现，但不影响 PADC 核心机制复现。

## 8. 当前机器条件

- OS：Windows；
- GPU：NVIDIA GeForce RTX 5060 Ti；
- GPU memory：16,311 MiB；
- Driver：591.86；
- 当前项目环境：
  - Python 3.13；
  - PyTorch `2.8.0+cu128`（运行时报告）；
  - CUDA available；
  - `scikit-image` 缺失。

建议使用 Python 3.10 的独立环境，并选择支持 RTX 5060 Ti/CUDA 12.8 的 PyTorch 构建；先做 checkpoint 兼容性测试，再冻结最终依赖。

## 9. 执行闸门

在以下顺序下推进，不跨级：

```text
R0 来源与环境审计
  ↓
R1 checkpoint 冒烟测试
  ↓
R2 DeepJSCC-V 从头训练
  ↓
R3 OraNet 从头训练
  ↓
R4 CIFAR 核心曲线
  ↓
R5 Kodak 与实例级 CR
  ↓
R6 正式复现报告
  ↓
开始 HAKE/HSC 创新
```

在 R6 完成前，不把 HAKE、HSC、OOD 或新的语义指标加入 PADC 代码。这样才能明确区分“原论文复现结果”和“我们的创新结果”。

