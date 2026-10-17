# SOTA 文献扫描与可复现代码核验（2026-09-15）

目的：在启动任何新实验之前，回答三个问题：
1. 语义通信（SemCom）相关方向 2025-06 至 2026-09 的最新工作是什么，SOTA 在哪里；
2. 哪些 SOTA 论文有**经核验**的公开代码/权重，可以在本机复现；
3. 我们现有的几条自有方法线是否应该现在启动"复现 SOTA + 改进"。

本文只做检索、核验与评估，**未启动任何复现或训练**。

---

## 0. 结论先行

**判定：暂不启动"我们的方法"，先启动"复现门"（R-gate）。**

- 我们当前三条自有线（有限重传调度 D55/D56、硬 2-bit ADC 接收机、KG-GAIL 452 主线）在最新文献中的定位都不利：
  要么被 2025-2026 顶刊工作正面撞题且对手无代码、无法公平比较（重传调度、KG/SKB），
  要么没有同任务对手可证 SOTA（2-bit ADC）。PADC 改造（R1）复现成功但 MMSE 改动增益
  `+0.0068 dB`、CI 含 0，且 PADC 是 2023 年方法，不适合作为 SOTA 起点。
- 经核验可复现的最强公开骨干是 **SwinJSCC**（TCCN 2024，官方仓库 200★，2025-09 仍在维护，官方权重可下载），
  它也是 2026 年数字/生成式 JSCC 论文最常用的骨干与基线（VQ-DSC-R、SSC 等）。
  2026 年唯一核验到"官方代码 + 顶刊录用"的是 **SSC（TCOM 2026，MIT，2026-09-06 仍在更新）**，
  但无预训练权重、依赖 BasicSR、信道层是小众的 FA-IM。
- 我们最感兴趣的可靠性/HARQ 方向（S3CHARQ、SPHARQ、SCB-HARQ、sem-HARQ ITS、proactive retransmission）
  **没有任何一篇核验到公开代码**。要在该方向工作，必须在 SwinJSCC 之上自行复实现基线并明确声明。
- 建议路线：R-gate 1 复现 SwinJSCC 官方权重（量化 torch 2.11 vs 论文 torch<=1.12 的差异）
  → R-gate 2 在其上复实现 2 个 HARQ 基线 → R-gate 3 建立公平基线表后再定义改进点。
  三道门未过之前，不写"我们的方法"。

---

## 1. Material Passport（检索过程与覆盖范围）

| 项目 | 内容 |
|---|---|
| 执行日期 | 2026-09-15 |
| 可用数据源 | OpenAlex API（4 组主题查询，2025-06-01 起，各 100 条 + 2024 起被引排序）；`arxiv.org/abs` 页面逐条抓取（21 篇）；GitHub REST API（10 组仓库搜索，其中 3 组因 SSL 中断失败；8 个仓库 README/目录核验，其中 4 个 README 抓取失败）；HuggingFace 模型搜索 |
| ���可用数据源 | WebSearch 工具与 `export.arxiv.org` API（持续 429 / SSL EOF）；`raw.githubusercontent.com`（阻断）；IEEE Xplore 全文（未尝试） |
| 原始产物 | `tmp/lit_scan_20260915.json`、`tmp/code_verify_20260915.json`、`tmp/repo_verify2_20260915.json` |
| 覆盖缺口（诚实声明） | (a) "semantic HARQ"、"DiffJSCC"、"DeepJSCC OFDM" 三组 GitHub 搜索失败，HARQ 方向可能漏掉个人仓库；(b) 仅在 IEEE 发表（无 arXiv 版）的论文无法核验代码链接，标为"未核验"；(c) S3CHARQ 全文本次未能重新抓取，方法描述来自其 arXiv 摘要与项目内 D55/D56 文档的既有阅读；(d) SwinJSCC 官方 README 本次 API 抓取失败，PyTorch>1.12 警告由官方仓库元数据 + 非官方复现仓库 README + 项目内 R1 协议记录三方交叉确认 |

OpenAlex 主题命中量（2024-01-01 起，作为热度参考）：

| 主题 | 命中数 |
|---|---|
| SemCom + JSCC + image | 238 |
| SemCom + HARQ/retransmission | 48 |
| SemCom + knowledge graph/base | 239 |
| SemCom + rate adaptation + image | 19 |

---

## 2. 三向扫描（WHY / HOW / WHAT）

### 2.1 方向 A：图像 JSCC 骨干（"SOTA 在哪"）

逐篇：

- **SwinJSCC**（Yang et al., TCCN 2024；官方 `semcomm/SwinJSCC`）
  WHY：CNN 型 DeepJSCC 在高分辨率与多 SNR/多码率下表现受限。
  HOW：Swin Transformer 编解码 + Channel ModNet（SA）与 Rate ModNet（RA），单模型多 SNR/多 CBR。
  WHAT：Kodak/CLIC21 上 PSNR/MS-SSIM 超 WITT/BPG+LDPC；官方权重公开。
  复现注意：官方与非官方复现仓库均指出 **PyTorch > 1.12 时性能低于论文**，非官方复现干脆去掉了 SA/RA 模块。

- **NTSCC / NTSCC+**（Dai/Wang et al., JSAC 2022 / JSTSP 2023；`wsxtyrdd/NTSCC_JSAC22`、`wsxtyrdd/NTSCC_plus`）
  WHY：JSCC 缺少可变速率的非线性变换与熵模型驱动的带宽分配。
  HOW：非线性变换 + 超先验熵模型决定各 patch 带宽，NTSCC+ 加入邻域注意力（NATTEN）与在线推理。
  WHAT：Kodak/CLIC 上优于 SwinJSCC 同期版本；权重公开，但 **NATTEN 锁定 0.14.6 + torch 1.9/cu111**，新版本 NATTEN 会给出错误结果。

- **SSC: Spatial Semantic Communication**（Guo et al., TCOM 2026，arXiv 2607.19934；官方 `gxh1106/SSC`，MIT）
  WHY：数字 SemCom 只兼容星座调制，未利用索引调制（IM）的额外维度。
  HOW：残差量化（RQ）把模拟语义特征离散化 + 语义感知流拆分（重要流走可靠子信道）+ 流体天线索引调制（FA-IM）；骨干 `SwinSSC_arch`。
  WHAT：RQ 保真、拆分提升可靠性、FA-IM 提升谱效；代码基于 BasicSR，DIV2K 训练/Kodak24 测试，**未提供预训练权重**，需从零训练。

- **VQ-DSC-R**（Chen et al., arXiv 2602.15045）
  WHY：数字 SemCom 多用简化信道。
  HOW：Swin 骨干 + 共享语义量化码本 + 自适应噪声方差可微 VQ + OFDM 三阶段训练。
  WHAT：高压缩比下鲁棒；**无代码**。

- **euJSCC**（Kim et al., arXiv 2602.14018）
  WHY：VQ-JSCC 的编解码器与码本固定于调制阶数，无法细粒度适应 SNR。
  HOW：超网络归一化 + 动态码本生成，单模型 SNR/调制自适应。
  WHAT：块衰落与 AWGN 下优于通道自适应数字 JSCC；**无代码**。

- 生成式一族：**FlowSem**（2608.21651）、**RC-BFM**（2607.24876，Globecom 2026）、**LGSemCom**（2609.06989，分组擦除信道）、**DiffJSCC**（TMLCN，`mingyuyng/DiffJSCC`，HF 权重但需 Stable Diffusion）、**SGDJSCC**（`MauroZMJ/SGDJSCC`，46★）。
  共同 HOW：接收端用扩散/流匹配先验补细节，牺牲延迟换感知质量；RC-BFM 专门压缩解码延迟（>10x）。
  仅 DiffJSCC/SGDJSCC 有代码。

- **SA-RA-JSCC**（2606.17940）：把 SNR 映射为统一语义向量做一次性全局重加权；无代码。
- **Ada-TokenCom**（2608.28086）：大模型 next-token 预测 + 算术编码的 token 级超低码率传输，Lyapunov 联合优化压缩率与 MCS；无代码。

**Common WHY**：单模型覆盖多 SNR/多码率，同时向数字调制兼容与感知质量靠拢。
**Divergent HOW**：模拟 JSCC 骨干（Swin/NTSCC）vs 离散化（VQ/RQ + 星座/索引调制）vs 生成式接收（扩散/流）vs token 级大模型。
**Strongest WHAT**：可复现且被广泛当基线的是 SwinJSCC 与 NTSCC+；2026 年有官方代码的顶刊工作只有 SSC。
**Unresolved gap**：数字化（VQ/RQ）、生成式接收与可靠性机制（HARQ）尚未在同一个可复现骨干上被联合评估；SwinJSCC 的 torch 版本敏感性没人系统量化。

### 2.2 方向 B：可靠性 / 语义 HARQ（我们 D55/D56 的方向）

- **S3CHARQ**（Li et al., arXiv 2603.23869，13 页）
  WHY：既有语义 HARQ 依赖额外校验码字只用于触发重传，开销大。
  HOW：把校验码字并入 JSCC（JS3C），校验码字同时做语义保真校验与重建增强；自适应重传。
  WHAT：相对既有 HARQ-SemCom，97th 百分位 PSNR +2.36 dB，中断概率 -37.45%；**无代码**（abs 页无 GitHub 链接）。

- **Toward Reliable Semantic Communication: Beyond Average Performance**（Li et al., 2606.01284，7 页综述）
  把可靠性设计分为三类：信道感知自适应、鲁棒编解码、HARQ 重传；明确列出未来方向：
  **鲁棒性与重传联合设计、超越均值的可靠性指标、与现有数字网络兼容**。

- **SPHARQ**（An et al., TWC 2026）：语义校验码（SCC）作语义失真即时检测代理 + 语义/物理指标联合驱动的动态重传准则 + 语义分组协作重传；代码未核验（IEEE-only）。
- **SCB-HARQ**（Liang et al., TCOM 2025）：收发共享离线语义码本作先验评估失真，WSFI 聚类降低码本传输开销；代码未核验。
- **sem-HARQ for ITS**（arXiv 2504.14615）：本地知识库支持的生成式信号重构器（判别器检错 + 条件生成纠错），提出 sem-HARQ-I / 加权合并 / 同义合并三种方案；无代码。
- **Proactive Retransmission-Based Image SemCom**（Yu et al., FCN 2025）：**发送端性能预测网络自主决定重传、无需反馈**，USRP B210 实测；无代码。
  注意：这与 PADC 的"传输前质量预测"思路结合重传的组合已被占位。
- **Tube-Structured Incremental Semantic HARQ**（2605.09897）：把"重传原语（primitive）"本身作为系统设计变量，生成式视频接收端在 AoIS-AUC 预算下驱动重传；无代码。
- **Generative Semantic HARQ (text)**（2603.15068，PIMRC 2026）：Transformer-VAE 随机编码器天然产生增量冗余，比较 6 种语义质量指标与 4 种软合并；无代码。
- **SITP**（TCOM 2026）：只校验包头、保留可能损坏的载荷交给语义解码，达到"TCP 可靠性 + UDP 延迟"，跨层丢包分析模型；无重传路线。
- 其它：HARQ + GAN 错检纠错（2025-12）、JS3C（Globecom 2025，S3CHARQ 前身）、多轮反馈语义 HARQ（PIMRC 2025）、量化噪声对抗 HARQ（2025-10）。

**Common WHY**：均值性能已够，尾部（低百分位 PSNR、中断概率）才是瓶颈；重传触发要用语义级校验而非比特级 CRC。
**Divergent HOW**：校验码字与 JSCC 联合（S3CHARQ）vs 独立语义校验码（SPHARQ）vs 共享码本先验（SCB-HARQ）vs 发送端预测免反馈（proactive）vs 接收端生成式纠错（sem-HARQ ITS）vs 改变重传原语（tube）。
**Strongest WHAT**：S3CHARQ 的尾部指标（97th PSNR、outage）是目前最硬的量化主张，但**无一篇可复现**。
**Unresolved gap**（综述作者自己列出的）：鲁棒性 + 重传联合设计、超越均值的可靠性度量、数字网络兼容。以上所有工作都在各自的自建/模拟 JSCC 上做，**没有在统一公开骨干（如 SwinJSCC）上的横向比较**。

### 2.3 方向 C：数字 SemCom（与我们 2-bit ADC/OFDM 积累相关）

- **MDJCM**（Zhang et al., TCOM 2024；`zhang-guangyi/Semantic-MDJCM`，基于 NTSCC，MDJCM-A 权重在百度网盘）：多阶数字联合编码调制。
- **Distribution-Aware Constellation Learning**（2605.30988）：从 APSK 几何先验出发的可学习星座 + 软分配/STE 两阶段训练；无代码。
- **Importance-Aware Constellation Design**（2605.14940）：VQ-VAE 离散概念 + 语义关键性指标 + DRL 选子集 + 语义感知 M-QAM 星座；无代码。
- **SSCDM**（2603.22983）：连续时间马尔可夫链构造的离散扩散做数字语义符号纠错；无代码。
- **PDSC**（2608.14260，Globecom 2026）：VLM 编码 + LDM 解码的个性化数字 SemCom；无代码。
- **DinoLink**（2606.26398，`UGA-MOBILITY-LAB/dino_link`）：token 级 RVQ 压缩用于 V2X 协同感知；有代码（未 API 核验）。
- **Adaptive Semantic Rate and Modulation via MoE**（CL 2026）；**Zero-Shot KB Resizing for Rate-Adaptive Digital SemCom**（WCNC 2026）：无代码。

**Common WHY**：与现有数字链路兼容。**Divergent HOW**：VQ/RQ 码本、可学习星座、索引调制、离散扩散纠错。
**Strongest WHAT**：有代码的只有 MDJCM（2024）与 SSC（2026）。
**Gap**：低比特 ADC/量化接收侧几乎无人做（这是我们 P3R 的原创点），但也意味着**没有同任务对手可以证明 SOTA**——这正是用户 2026-09-15 判定该线过窄的原因。

### 2.4 方向 D：KG / SKB SemCom（我们的 452 主线）

2026 年新工作：**SSKBQ 知识库复用**（2609.10112）、**本体驱动 SemCom**（2608.25422）、
**云边协同个性化 KB 部署**（Sensors 2026-08）、**Neurosymbolic SKB 重要性加权推理 + 增量更新**（IoT-J 2026-07）、
**KB 识别与自适应**（IoT-J 2026-07）、**GLASS KG-guided LLM-assisted**（WiOpt 2026）、
**Future-Aware Cognitive SemCom via KG+LLM**（ICC Workshops 2026）、**HSCKBase 分层联邦 SKB**（TMC 2026）、
**SC-LMKB LLM 生成 KB**（2604.05504）。高被引锚点：Liang TCCN 2024（139 引）、Jiang LAM 系列（110+）、Ren WCM 2024 KB-enabled（58 引）。

**Common WHY**：收发端知识不同步；**Divergent HOW**：LLM 生成/补全 KB vs 本体规则 vs 联邦更新 vs 云边部署优化 vs 神经符号增量更新。
**Strongest WHAT**：没有一篇核验到公开代码；评价指标各自为政。
**评估**：我们 E8 的 `incremental_align` 与"云边个性化 KB 部署"和"Neurosymbolic SKB 增量更新"直接撞题，且对手不可复现；领域正被 LLM-KB 吞并，而本项目硬约束禁止 LLM API。**不适合作为"复现 SOTA + 改进"的起点。**

---

## 3. 可复现代码核验表

核验方式：GitHub API 仓库元数据 + README 全文（或非官方镜像 README）+ 本地环境包检查。"权重"指 README 明确给出下载链接。

| 方法 | 发表 | 仓库 | ★ | 最近 push | 权重 | 环境钉死 | 本机风险 | 核验状态 |
|---|---|---|---|---|---|---|---|---|
| SwinJSCC | TCCN 2024 | `semcomm/SwinJSCC` | 200 | 2025-09-14 | 有（百度盘） | 官方警告 torch>1.12 性能下降 | 本机 torch 2.11：需先量化差异 | 元数据核验；README 经非官方镜像交叉确认 |
| SwinJSCC 非官方重构 | — | `serika1108/SwinJSCC` | 6 | 2025-12-09 | 可加载官方权重（strict=False） | 去掉 SA/RA | 非官方，不能当基线 | README 全文核验 |
| NTSCC | JSAC 2022 | `wsxtyrdd/NTSCC_JSAC22` | 109 | 2023-07-15 | 有（GDrive，仅 w/o z） | torch 1.7.1+cu110 | 作者注明与论文略有差异 | README 全文核验 |
| NTSCC+ | JSTSP 2023 | `wsxtyrdd/NTSCC_plus` | 30 | 2025-01-03 | 有（GDrive） | natten==0.14.6+torch190cu111，compressai/timm | RTX 5060 Ti（cu128）无法装老 natten wheel，需独立老环境或放弃 | README 全文核验 |
| MDJCM | TCOM 2024 | `zhang-guangyi/Semantic-MDJCM` | 29 | 2024-12-17 | 有（百度盘，仅 MDJCM-A） | Python 3.9.12；"逐步上传完整版" | 完整性未知 | README 全文核验 |
| SSC | TCOM 2026 | `gxh1106/SSC`（MIT） | 2 | 2026-09-06 | 无，需 DIV2K 从零训练 | BasicSR 本地安装 | 训练时长与 FA-IM 小众性 | README + 目录核验 |
| WITT | ICASSP 2023 | `KeYang8/WITT` | 214 | 2024-08-27 | 有（百度/GDrive） | Python 3.8 / torch 1.9 | 本地 zip 不含权重 | README 全文核验 |
| DiffJSCC | TMLCN 2024 | `mingyuyng/DiffJSCC` | 49 | 2025-07-14 | 有（HuggingFace） | 需 Stable Diffusion 权重 | 显存/延迟大 | README 全文核验 |
| SGDJSCC | 2024 | `MauroZMJ/SGDJSCC` | 46 | 2025-04-01 | 未核验 | — | — | 仅搜索列表 |
| Dynamic-JSCC | ICASSP 2022 | `mingyuyng/Dynamic_JSCC` | 152 | 2022-09-21 | 本地有 `Checkpoints/` | 旧 CycleGAN 风格 | 已是 2022 年方法 | 本地副本核验 |
| OFDM-guided JSCC | TCCN 2021 | `mingyuyng/OFDM-guided-JSCC` | 80 | 2021-09-14 | 本地仅 `Pilot_bit.pt` | — | 无模型权重 | 本地副本核验 |
| PADC (DeepJSCC-V + OraNet) | TCCN 2023 | 本地 | — | — | 有（本地 .pth.tar） | — | R1 已复现 | 本项目 R1 |
| DinoLink | 2026 | `UGA-MOBILITY-LAB/dino_link` | ? | ? | ? | — | 非图像重建任务 | 仅 abs 页链接 |

**明确无代码（abs 页核验）**：S3CHARQ、Beyond-Average 综述、Tube HARQ、Generative Semantic HARQ、FlowSem、RC-BFM、LGSemCom、SA-RA-JSCC、Ada-TokenCom、Distribution-Aware Constellation、euJSCC、VQ-DSC-R、SSKBQ、SC-LMKB、Ontology SemCom、PDSC、sem-HARQ ITS。
**无法核验（IEEE-only）**：SPHARQ、SCB-HARQ、SITP、CG-SemCom、Neurosymbolic SKB、KB Identification、HSCKBase、MoE 速率调制、Zero-shot KB resizing。

本机环境（核验）：torch 2.11.0+cu128，RTX 5060 Ti；缺 timm / compressai / natten / scikit-image / pytorch_msssim / basicsr / einops。
本地数据：CIFAR-10/100/10.1；无 DIV2K / Kodak / CLIC。

---

## 4. 我们的方法线逐条评估

| 线 | 当前状态 | 文献定位 | 判定 |
|---|---|---|---|
| 有限重传调度 + 增益预测（D55/D56） | D55 已执行，D56 预测器未过门槛 | 被 S3CHARQ / SPHARQ / SCB-HARQ / proactive retransmission 正面覆盖；"预测后决定重传"已被 FCN 2025 占位；我们跑在自建 32x32 冻结 codec `StrongSpatialSemanticOFDM` 上，无外部可比性 | **不启动**，除非先在 SwinJSCC 上复实现对手 |
| 硬 2-bit ADC 语义 OFDM 接收（P3R 主线） | D34C 整体 FAIL（10/11），用户 2026-09-15 判定过窄 | 无同任务对手；无法自证 SOTA | **封存**，作为可能的"数字兼容"子模块保留 |
| KG-GAIL 452 主线 + E8 云边 SKB 对齐 | 完整但冻结 | 2026 年撞题工作全部无代码；领域转向 LLM-KB，与硬约束冲突 | **不作为起点** |
| PADC 改造（R1 MMSE） | 复现通过，改动增益 +0.0068 dB，CI 含 0 | PADC 为 2023 年方法，OraNet 思路已被 proactive retransmission 组合使用 | **不再叠加模块**；OraNet 式"传输前质量预测"可作为未来改进点的组件之一 |

---

## 5. 建议路线（门控，不过门不写方法）

| 门 | 内容 | 通过标准 | 估计工作量 |
|---|---|---|---|
| R0 环境 | 独立 conda 环境装 SwinJSCC 依赖；下载官方权重、Kodak24、CLIC21；准备 DIV2K（若需重训） | 官方 `main.py` 测试模式跑通 | 0.5-1 天 |
| R1 SwinJSCC 复现 | 官方权重在 Kodak/CLIC 上按论文设置（AWGN/Rayleigh，SNR 1/4/7/10/13，CBR 若干）评测 PSNR/MS-SSIM | 与论文表格逐点差值记录；若 torch 2.11 差异 >0.3 dB，尝试 torch 1.12 环境复测并写入差异报告 | 1-2 天（评测）；若需重训 3-7 天 |
| R2 基线复实现 | 在 R1 骨干上复实现 2 个可靠性基线：(a) S3CHARQ-lite（校验码字并入 JSCC + 语义校验触发重传），(b) SCB-HARQ-lite 或 proactive-retransmission-lite（发送端预测触发）；统一尾部指标（5th/97th 百分位 PSNR、outage、平均重传次数、有效 CBR） | 论文中可对照的趋势方向一致（如 S3CHARQ 的尾部增益为正）；全部标注为 reimplementation | 1-2 周 |
| R3 定义改进点 | 仅在 R2 基线表上仍存在明确 gap 时定义 | 改进点必须对应第 2.2 节综述列出的未占位方向之一 | — |

R3 的候选缺口（先验证未被占位再选，**不是现在就做**）：
1. **鲁棒性 + 重传联合设计**：现有 HARQ 论文把 JSCC 骨干当黑盒；在 SwinJSCC 上联合训练"重传感知"的编码器（增量冗余由编码器结构产生，而不是重复发送）。
2. **数字兼容的语义 HARQ 尾部保证**：把 SSC/MDJCM 式 RQ/VQ 离散化与语义 HARQ 结合，用 97th 百分位 PSNR/outage 作主指标；需先确认 SCB-HARQ 未覆盖（其为码本先验检测，非数字调制链路）。
3. **重传原语选择**（tube 论文提出但只在视频上做）：图像 patch/流级重传单元 vs 整帧，配合 SwinJSCC 的 patch 级速率分配。

明确不建议：继续给 `StrongSpatialSemanticOFDM` 叠模块；在无对手代码的 KG/SKB 方向自证 SOTA；用 NTSCC+ 作主骨干（NATTEN 版本锁死，本机 GPU 装不了）。

---

## 6. 留给用户的决定（非阻塞）

1. 主骨干选 SwinJSCC（权重现成、社区最大）还是 SSC（2026 最新、需从零训练、FA-IM 小众）？本报告建议 SwinJSCC。
2. 是否接受"对手基线为本项目复实现"这一论文写法（HARQ 方向无官方代码，别无选择）？
3. 是否需要为 R1 建一个 torch 1.12 的老环境来排除版本差异（额外 0.5 天）？
