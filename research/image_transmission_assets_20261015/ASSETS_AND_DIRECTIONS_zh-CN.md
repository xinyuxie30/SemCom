# 图像传输：现有资产、模块来源与候选研究问题

日期：2026-10-15。范围：静态图像重建与传输，信道作为可控条件；本地仿真，无 LLM/LAM API。本文是资产盘点与定向文献综合，不是完成查新的创新声明，也不是新实验结果。

本轮实际工作：查阅本地源码、报告与审计文件；从 PADC 原始 CSV 重算主条件均值；核对 B0.1 汇总；清点 DIV2K；核查作者仓库和论文页面。未启动训练、GPU 推理或旧封存测试。检索范围与证据声明见 [evidence_scope.json](evidence_scope.json)。

**我的建议：以“图像重建质量—通信量—实际计算时延”为主坐标。先验证 FAJSCCv2 中的接收端计算分配是否还有可利用空间；DCAT 用于已有信道失配问题的短期诊断；渐进补传作为第二候选。**

这里的“优先”是按现有资产和验证成本排序，不代表已证明新颖或有效。FAJSCCv2 正式基线尚未完成，DCAT 的训练入口也不完整；二者的可复现成熟度不同。

**1. 如何采用那位同学的方法**

值得保留的是：读经典、逐层调试、记录张量与机制、建立模块卡、在强基线上做公平消融。

需要修正两点：

- 改模块名字不会产生新贡献。复用应注明原始论文、源码和修改位置；新的名字只能用于确实改变了功能的实现。
- 论文不要求硬凑三个独立创新点。“一个明确问题 + 一个有效机制 + 充分验证”比三个拼装模块更完整。不能先看到偶然涨点，再事后挑一个没有证据的解释。

合理顺序是：**观察具体失败 → 写可证伪假设 → 找能解决这个失败的模块 → 预先固定对照 → 看增益是否独立存在**。一次只改变一个主要机制。

**2. 我们真正已有的资产**

证据等级：A = 本地有实际运行与可追溯结果；B = 仅最小运行通过；C = 本地有源码/压缩包但本轮未发现该原版的完整运行证据；D = 本轮核查的外部候选。A 也不自动代表论文全部复现或独立泛化确认。

| 资产 | 本地状态与证据 | 可以直接复用什么 | 不能扩大解释为什么 |
|---|---|---|---|
| DCAT 官方权重及本地评测层 | A：225 条件、2,212 图、497,700 次传输；单噪声种子；5 个检查点 | 图像编解码、DC-attn、DC-permu、时变信道、批处理和指标账本 | 只完成公开权重 Case A 推理；未完整复现训练及所有对手；与旧预印本有数值差异 |
| PADC / DeepJSCC-V / OraNet | A：3 个作者检查点严格加载，12 个合成保真检查；172 图、13,932 行开发比较 | 变码率 codec；传输前质量预测；同噪声均衡器比较 | 尚未完成原文 CIFAR/Kodak 曲线；不是当前 SOTA 证明 |
| FAJSCCv2 | B：作者训练入口完成一次更新、保存、重载；AWGN/Rayleigh 小输入推理通过 | AttentionTree、FAmixer、选择性可变形注意力、编码/解码独立算力比例 | 一张合成图、batch=1；没有有效正式模型或论文性能复现证据 |
| Dynamic-JSCC | C：本地源码与训练入口 | 动态速率选择、软/硬选择的训练方式 | 源码存在不等于本机当前版本已复现 |
| OFDM-JSCC | C：本地源码；另有 P3R 自建仿真资产 | 可微 OFDM、导频、估计/均衡、残差接收模块思路 | 本地文件夹名称与 README 指向的论文版本需要区分 |
| WITT | C：本地 PDF 与 ZIP | Swin 图像编解码与信道调制的入门参考 | 本轮未发现该原版的当前完整运行记录 |
| DCAT 因果预测实验层 | A：hold/AR/Wiener/ACF、因果性检查、配对评测、校准与消融 | 可观察 CSI 接口、训练/验证时间划分、错误分层 | 当前直接替换方案没有过图像质量门槛 |
| 两轮图像反事实评测层 | A：384 状态的小试；完整流水线另有 smoke | 统一状态下枚举动作、质量/资源/等待统计、可部署输入与真值分离 | 当前裁剪重编码与固定融合失败；整套候选方法未完成正式实验 |
| P3R 历史工程 | A：有受限场景正式结果，也有被判无效的历史分支 | 物理预算账本、逐帧不变性检查、配对噪声、审计与统计方法 | 低比特接收机路线已停止；D17–D40 有批内信息混合，不能当有效部署证据 |
| KG / GAIL / SKB | 有历史代码和实验记录；本轮未重新审核其效果 | 本地仿真和实验管理经验；将来确有知识传输需求再考虑 | KG 推理成功率不是图像质量证据，不应强接入图像主干 |

主要本地证据：

- [DCAT 推理报告](D:/PyChram_project/DCAT-SoM/results/20260920_232752_published_inference_suite/REPORT.md)。<!--ref:local_dcat_report--><!--anchor:section:结论-->
- [PADC R1 结果](D:/PyChram_project/CognitiveSemCom/reproduction/padc/R1_RESULTS_20260915_zh-CN.md)。<!--ref:local_padc_r1--><!--anchor:section:原版保真核验-->
- [FAJSCCv2 运行报告](D:/PyChram_project/CognitiveSemCom/reproduction/FAJSCCv2_run_check_2026-10-07/RUN_REPORT.md)。<!--ref:local_fajscc_run--><!--anchor:section:实际结果-->
- [P3R 当前状态](D:/PyChram_project/CognitiveSemCom/P3R_SemCom/docs/ACTIVE_RESEARCH_STATUS_20260915_zh-CN.md)及[分支有效性约束](D:/PyChram_project/CognitiveSemCom/P3R_SemCom/AGENTS.md)。<!--ref:local_p3r_status--><!--anchor:section:当前研究状态：已换方向，尚无新%20SOTA-->

数据也已经构成资产：DIV2K 800 张训练原图与 100 张验证原图已备齐，本轮清点数量一致；DCAT 有 2,212 张 UDIS-D 评测输入；Kodak 已用于先前小试。后两者已有开发使用历史，不可重新称为“完全未见测试集”。新协议需明确训练、开发、校准和最终评估的用途。[DIV2K 准备记录](D:/PyChram_project/CognitiveSemCom/reproduction/FAJSCCv2_run_check_2026-10-07/formal_reproduction/DATASET_README.md)。<!--ref:local_div2k--><!--anchor:section:数据位置-->

**3. 已经获得、值得留下的结论**

| 观察 | 证据 | 对下一步的实际含义 |
|---|---|---|
| CSI 更准不自动等于图像更好 | B0.1 的 A7：接收端 NMSE 0.67409 → 0.22061，下降约 67.27%；ACF+校准相对原 hold 为 −0.03754 dB，相对同样校准的 hold 仅 +0.00436 dB | 不能仅凭 NMSE 决定扩训预测器；先检查解码器和失真分布是否匹配 |
| 当前两轮 codec 效率不足 | C128+C64 的逐状态 oracle 为 24.706 dB，低于等载荷单轮 C192 的 25.602 dB；尚未扣额外控制成本 | 换选择器不能解决这个固定动作集的问题；需要先让新增传输带来有效新增信息 |
| 直接预测收益未必比质量预测更好 | D56 六种条件均未达到相对同容量质量预测的晋级要求 | 新方案必须加入有依据的新信息或改变决策对象，不能只换监督标签 |
| 经典接收机改动的效果依赖 codec | PADC 主条件 MMSE +0.22919 dB，但未达原 +0.3 dB 门槛；DCAT 冻结模型下当前 MMSE 插入普遍退化 | 保留经典方案作为强对照；不能宣称 MMSE 必然改善或必然无效 |
| 工程证据也是资产 | 固定代码/权重哈希、相同样本及噪声、资源账本、独立重算已经做过 | 新方向应继承这些规范，减少虚假增益 |

来源：[B0.1 审查](D:/PyChram_project/CognitiveSemCom/research/dcat_csi_prediction_20261008/results/b01_20261008T004209Z/review/REVIEW_zh-CN.md)、[两轮小试](D:/PyChram_project/CognitiveSemCom/research/jev_image_transmission_20260926/PILOT_RESULTS.md)、[D55/D56](D:/PyChram_project/CognitiveSemCom/P3R_SemCom/docs/D55_D56_PIVOT_RESULTS_20260915_zh-CN.md)。<!--ref:local_b01--><!--anchor:section:3.%20主结果：固定%20without_CP_C128--><!--ref:local_roi_pilot--><!--anchor:section:冻结%20codec%20的选区空间--><!--ref:local_d56--><!--anchor:section:新假设未通过-->

这些主要是固定开发条件下的结果。B0.1 的三个种子只改变加性噪声，并非三个训练种子或三条独立传播轨迹。

必须纠正旧笔记中的两处冲突：

- 10 月 4 日规划把官方 with_CP 当成 hold，不准确。官方推理的 with_CP 使用真实 CSI、零年龄及另一套权重，不能把两权重族之差当作纯因果预测改进空间。此次据源码及 10 月 8 日审查采用修正后的解释。
- 9 月 15 日扫描中的 PADC “+0.0068 dB”与 R1 主记录不一致。本轮由原始 13,932 行 CSV 重算主条件为 28.328997−28.099806 = **+0.229191 dB**；本文使用可追溯的原始记录。

**4. 经典论文应该怎样读、怎样抽模块**

以下是阅读顺序，不是要求把全部网络训练一遍。每一篇都要回答：输入是什么、哪个张量过信道、预算怎么计、模块改变了什么、移除后发生什么。

| 顺序 | 论文与版本 | 最值得学的机制 | 我们的状态/入口 |
|---|---|---|---|
| 1 | Bourtsoulatze 等，DeepJSCC，TCCN 2019 | 图像直接映射复符号；功率归一化；端到端失真训练 | [原论文](https://arxiv.org/abs/1809.01733)；先掌握基础链路 |
| 2 | Kurka、Gündüz，DeepJSCC-f，JSAIT 2020 | 反馈条件编码、累积接收信息、可学习输出融合 | [作者代码](https://github.com/kurka/deepJSCC-feedback)；旧 TensorFlow 路线，未在本机复现 |
| 3 | Yang 等，OFDM-JSCC，ICC 2021；Dynamic-JSCC，ICASSP 2022 | 物理信道在环与自适应速率 | [OFDM 代码](https://github.com/mingyuyng/Deep-JSCC-for-images-with-OFDM)、[Dynamic 代码](https://github.com/mingyuyng/Dynamic_JSCC)；本地已有 |
| 4 | Dai 等，NTSCC，JSAC 2022；Wang 等，NTSCC+，JSTSP 2023 | 超先验/概率模型引导局部带宽分配 | [NTSCC](https://github.com/wsxtyrdd/NTSCC_JSAC22)、[NTSCC+](https://github.com/wsxtyrdd/NTSCC_plus)；外部待验证 |
| 5 | Zhang 等，PADC，TWC 2023 | 编码特征统计 → PSNR 预测 → 速率选择 | [作者代码](https://github.com/wyzhang-ustb/Predictive-and-Adaptive-Deep-Coding-for-Wireless-Image-Transmission-in-Semantic-Communication)；优先逐行 debug |
| 6 | WITT，ICASSP 2023；SwinJSCC，TCCN 2024 | Transformer 图像表示；信道/码率条件调制 | [WITT](https://github.com/KeYang8/WITT)、[SwinJSCC](https://github.com/semcomm/SwinJSCC) |
| 7 | Zhang 等，HJSCC，AAAI 2025 | 层级潜变量与反馈扩展 | [论文出版页](https://ojs.aaai.org/index.php/AAAI/article/view/33442)、[作者代码](https://github.com/zhang-guangyi/HJSCC) |
| 8 | FAJSCCv2；DCAT | 选择性计算；时变信道适配 | [FAJSCCv2](https://github.com/hansung-choi/FAJSCCv2)、[DCAT](https://github.com/PKU-PCNI/SoM-DCAT)；近期主参考 |

<!--ref:deepjscc--><!--anchor:section:Abstract--><!--ref:deepjscc_f--><!--anchor:section:README--><!--ref:ofdm--><!--anchor:section:Reference--><!--ref:dynamic_jscc--><!--anchor:section:README--><!--ref:ntscc--><!--anchor:section:README--><!--ref:ntscc_plus--><!--anchor:section:Requirements--><!--ref:padc--><!--anchor:section:README--><!--ref:witt--><!--anchor:section:Citation--><!--ref:swinjscc--><!--anchor:section:Introduction--><!--ref:hjscc--><!--anchor:section:Abstract--><!--ref:fajscc--><!--anchor:section:README--><!--ref:dcat--><!--anchor:section:Overview-->

版本边界：HJSCC 出版年份是 2025，不能照抄仓库 BibTeX 的 2024 为会议年份；WITT 是 ICASSP，不是 ICLR；PADC 是 TWC，不是旧扫描表误写的 TCCN。FAJSCCv2 README 链接 IoT-J 论文，DCAT README 链接 2026 TWC 版本；本文对方法细节仍区分公开预印本与期刊版本。

NTSCC+ 明确要求旧 NATTEN 版本，且在线推理 NTSCC++ 在 README 仍为 TODO；SwinJSCC 官方提醒 torch>1.12 下 SA/RA 推理存在结果不一致。因此“有代码、列出权重链接”只算候选条件，不算本机已可复现。DeepJSCC-L++ 的单模型多带宽/多 SNR 与动态损失加权也值得作为训练参考，但不应误认成原生多轮渐进协议。[L++ 作者代码](https://github.com/aprilbian/deepjscc-lplusplus)。<!--ref:deepjscc_lpp--><!--anchor:section:Overview-->

**5. 首选研究问题：把接收端算力分给真正能改善重建的位置**

候选题目：**面向有限解码预算的重建收益感知图像传输**。

RQ：固定发送端、载荷和平均功率，在相同实际解码时延预算下，依据接收特征与可观察失真可靠性，预测某个窗口“再执行一次较贵的计算能降低多少图像误差”，是否优于 FAJSCCv2 原来的重要性分配？

直观例子：复杂纹理可能很重要，但已恢复得很好；一个受损边缘可能通过一次注意力计算就能修复；另一个严重衰落区域即使投入计算也未必恢复。应学习的是**新增计算的重建价值**，而不只是纹理复杂度、失真大小或信道好坏。

推荐组合：

1. FAJSCCv2 的选择性计算结构提供“算哪里”的执行能力。
2. 受 OraNet 启发的小预测头提供“多算的边际收益”，输入改为接收特征与可部署可靠性信息；不直接搬用其 PSNR 权重。
3. 原有信道与统计工具提供可控退化、配对比较和资源记录；第一版不需要额外反馈，也不需要强化学习。

概念接口为：接收潜变量 y、可靠性特征 u、预算 b → 每窗口收益估计 g → 执行集合 S → 图像重建。可靠性可以先用所有方法共同可用的 SNR/接收特征，再考察通过因果导频估计得到的局部信息；不能读取真实误差或未来 CSI。

训练期可以在同一接收状态下，比较“某窗口经过重计算”和“仅走轻分支”的整图失真差：

Δ_j = D(x, x_hat_light) − D(x, x_hat_with_j)。

这是候选监督定义，尚未实现。局部动作会相互影响，不能假定 Δ_j 可简单相加；第一轮只在一个解码 stage 做少量候选集合的枚举或单窗口干预，把它作为有限动作集诊断，而非全局最优上界。

**可能形成的贡献，均需实验支持：**

- 问题与证据：量化“原始内容重要性”与“受损后新增计算的价值”的偏离，以及它在哪些图像/信道/预算条件下出现。
- 方法：面向该价值的接收端路由，建立收益监督、可部署输入与硬计算预算之间的对应。
- 系统结果：在同载荷、同能量、同真实时延下，改善重建质量—计算成本曲线，并检验最差分位质量。若只有第一、二点成立，不必硬塞第三个模块。

**与前作的界限。** FAJSCC 已经是重要性感知和可调计算，而且其接收特征隐含了噪声信息，不能称其“完全不感知信道”。新方案必须证明显式的条件收益估计比其已有可训练路由更好；仅增加 SNR、改 loss 名字或替换注意力类型都不够。U-DeepSC 也已有多出口机制，不能宣称首个语义通信早退出。[FAJSCC v2 方法](https://arxiv.org/html/2504.04758v2)、[U-DeepSC](https://arxiv.org/abs/2206.00254)。<!--ref:fajscc--><!--anchor:section:III%20FAJSCC%20Framework--><!--ref:udeep--><!--anchor:section:Abstract-->

**最便宜的决定性实验：**

- 先完成有效的 FAJSCCv2 原版训练/验证。当前合成图权重不能用于判断研究空间。
- 冻结同一个可用 backbone，在固定 stage、固定窗口数和配对噪声下比较：原路由、随机路由、纹理/能量启发式、仅可靠性路由、训练期真值指导的候选集合选择。
- 如果真值指导也几乎没有改善，停止这个路由方向；如果有空间，再训练小头。真值仅生成训练标签和诊断上界，部署时移除。
- learned 对照至少包含“原重要性头继续训练”“同输入同容量普通质量头”“收益头”“收益头+可靠性”，训练步数及数据一致。D56 已经说明只改目标可能没有价值。
- 预先选定主门槛，例如同真实时延平均 PSNR 增益 ≥0.2 dB，同时 pooled MSE、LPIPS 与差图像分位不恶化。0.2 dB 是建议的开发筛选门槛，不是预测结果或统计定律。正式推断须再固定独立数据和训练种子。
- 所有计时包含预测头、排序和 gather/scatter；先测 batch=1。FLOPs 更少但实际更慢不能叫低时延改进。

首个工程缺口：FAJSCCv2 原源码的评估路由仍有 Gumbel 采样，保留窗口数还使用跨 batch 聚合；信道将整个 batch 展平后配对实虚分量，MMSE 的复噪声方差口径也需要核查。这些是本轮静态阅读定位的问题，不是已完成的故障复现实验。保真复现与修正版应分开记录，并让所有新旧比较方法使用同一已核验信道。

**6. 第二候选：用已有 DCAT 负结果研究图像潜变量修复**

RQ：固定因果预测器，加入等容量的轻量解码适配，能否让“CSI 误差降低”真正转化为“图像误差降低”？

可用 DCAT + 小型残差模块，NAFNet 的残差块是一个代码清楚的图像恢复组件参考。建议先在恢复顺序后的潜变量与解码器之间尝试，比较无条件适配、SNR 条件适配、误差结构条件适配；hold 和预测器分支都给同样训练预算。[NAFNet 作者代码](https://github.com/megvii-research/NAFNet)。<!--ref:nafnet--><!--anchor:section:README-->

这是一项便宜的机制诊断，优先于扩训 GRU/TCN，但未必能成为独立论文。DCAT 本身已预测 NMSE 并调制网络；HANA-JSCC 已有不完美 CSI 的适配与蒸馏，TokenComSR 已有 SNR 条件 token 残差修复。仅“加适配器/去噪器”重叠很大。[HANA-JSCC](https://arxiv.org/abs/2509.04005)、[TokenComSR](https://arxiv.org/abs/2609.03735)。<!--ref:hana--><!--anchor:section:Abstract--><!--ref:tokencomsr--><!--anchor:section:Abstract-->

可能研究的差异是：真实因果预测产生的有偏、时序相关误差，而不是人为独立噪声；是否需要保留误差结构，才能恢复下游质量。这个解释仍是待检验假设。沿用已有 A7 +0.3 dB、A1 回退≤0.05 dB 的开发门槛时，应比较最强同容量 hold，且同时核查 MSE、LPIPS 与尾部；失败就收口。

**7. 第三候选：新增传输应该提供新增信息**

RQ：在同总资源约束下，基于基础层状态的条件增强编码与可拒绝坏增强的融合，能否减少“传了更多却重建更差”的图像比例？

模块来源：HJSCC 的层级表示或 DeepJSCC-f 的反馈编码/融合；PADC 式质量头只能作为辅助；本地反事实评测和资源账本可以直接复用。

第一步不做四象限选择和复杂调度，先做全局基础层+一个条件增强层。编码器若使用接收端首轮重建，必须说明反馈如何得到并计入开销；无反馈版本只能用发送端可知的基础表示，不能免费读取接收端随机噪声后的重建。

必须先过 codec 门：两层方案在预先固定的质量—总资源比较上达到单轮强基线，再研究按图像停止或区域分配。接收融合保留基础图作为回退选项，但没有测试原图的接收端不能保证每张图都不退化；应报告退化概率和幅度，不写数学单调保证。

渐进编码、层级 VAE、反馈增强、重要性分配均已有前作，包括 DeepJSCC-l、JSCCformer-f、HJSCC、FAST；近期 TS-JSCC 已有内容自适应前缀传输。需要查清坏增强拒绝、相关衰落、有限反馈方面的直接工作，不能把简单组合当首创。[DeepJSCC-l](https://arxiv.org/abs/2009.12480)、[JSCCformer-f](https://arxiv.org/html/2306.09101v3)、[FAST](https://arxiv.org/abs/2401.14614)、[TS-JSCC](https://arxiv.org/abs/2608.21743)。<!--ref:deepjscc_l--><!--anchor:section:Abstract--><!--ref:jsccformer_f--><!--anchor:section:Abstract--><!--ref:fast--><!--anchor:section:Abstract--><!--ref:tsjscc--><!--anchor:section:Abstract-->

**8. 不值得现在继续堆的组合**

- 更大的 CSI 预测器 + 原封不动的冻结 DCAT：已有不通过的端到端证据。
- 当前裁剪放大重编码 + 更复杂的选择器：固定动作集的诊断上界未过单轮。
- 任意 Transformer + CBAM/SE + “语义感知”的改名：没有指出新问题和独立机制。
- 仅重要性/熵排序 + 信道分配：FAST、NTSCC、DCAT、TokenComSR 等已覆盖大量相邻机制。
- 把旧 KG/GAIL、低比特 ADC 和图像主干同时合并：扩大变量，不直接解释图像增益。

这不否定模块本身。已知模块适合当组件和强基线；需要证据的新意是它们解决了什么尚未解决的问题。

**9. 最小模块仓库与阅读安排**

本次建立的是模块索引和卡片，未复制外部源码。每张卡记录：原始来源/版本、输入输出、插入点、预期作用、成本、已知失败、最小对照。详细位置见 [MODULE_CARDS_zh-CN.md](MODULE_CARDS_zh-CN.md)。

建议顺序：

1. PADC：一张图从 encoder → 变码率掩码 → 归一化 → 信道 → decoder → OraNet，逐步打印张量、功率和真实复符号数。
2. FAJSCCv2：同一张图改变解码计算比例；记录选择窗口、输出差异与真实时间，检查 batch 和随机性。
3. DCAT：固定同检查点，跟踪 CSI 预测怎样改变重排、均衡和条件调制，定位图像退化出现在哪一段。
4. 仅当确定研究渐进传输时，补读 HJSCC/DeepJSCC-f；仅当做局部码率分配时，补读 NTSCC。

所有新对照统一：输入与分辨率、每图复信道使用量、实际平均功率、可用 CSI、训练数据与预算、噪声配对、指标输入范围。PSNR 与 pooled MSE 分开；LPIPS 用同一归一化口径；增加导频、反馈或控制字段时全部记账。已有开发数据不包装成最终盲测。

**10. 查新与解释边界**

本轮有意优先选择作者代码和一手论文；没有枚举所有期刊、引用链或所有动态图像恢复方法。发现了足以否定宽泛新颖性主张的近邻，尚未完成首选问题的全文排重。

补充修正：旧扫描中“可靠性方向都没有公开代码”的表述不能沿用；本轮找到 [SC-TDA-HARQ 作者实现仓库](https://github.com/nf129/SC-TDA-HARQ)。目前只核查到仓库说明，未验证权重、训练或论文级可复现性，不将其列为已跑通资产。<!--ref:sc_tda_harq--><!--anchor:section:README-->

跨来源关系的有限检查如下；研究者确认状态均为 pending，不声称穷尽所有冲突：

| 来源对 | 关系判断 | 如何处理 |
|---|---|---|
| FAJSCC 与 DCAT/B0.1 | 无实质矛盾：前者分配计算，后者处理信道失配 | 由二者提出条件计算价值问题，是推断，不是前作结论 |
| PADC MMSE 与 DCAT MMSE | 条件差异：codec、CSI 与适配状态不同 | 保留两种方向的结果，不能给 MMSE 普适正负结论 |
| HJSCC/DeepJSCC-f 与本地裁剪补传 | 条件差异：联合/条件编码不同于独立裁剪重编码 | 本地负结果只否定当前 codec/动作集，不否定渐进传输 |

当前可以负责任地提出的是**三个可验证的研究问题**。首选问题能否成为创新，取决于 FAJSCC 的匹配比较、额外可利用空间、真实时延收益，以及补齐近邻全文比较。
