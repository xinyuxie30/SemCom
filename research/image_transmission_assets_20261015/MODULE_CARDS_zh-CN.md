# 图像传输模块卡与逐步调试路线

2026-10-15。此文件是可追溯的模块索引，不是已经抽取、集成或验证的新代码库。外部模块作为明确引用的机制来源；复制实现前固定提交和许可证，移植后的实现单独记录。本轮没有训练新方法。

优先级：P0=当前优先理解与验证；P1=选定对应问题后使用；P2=参考/强对照。不要同时把所有模块接入一个网络。

**M1｜FAJSCCv2：选择性计算，P0**

- 来源：[作者代码](https://github.com/hansung-choi/FAJSCCv2)、[v2 论文](https://arxiv.org/html/2504.04758v2)，本地固定提交 99987dcb75e46ce3ac754d8854dd3f04cac5a1b8。
- 核心代码：[AttentionTree / FAmixer / FABlock](D:/PyChram_project/CognitiveSemCom/reproduction/FAJSCCv2_run_check_2026-10-07/upstream/model/FAComponent.py:139)。
- 模型入口：[FAJSCC.forward](D:/PyChram_project/CognitiveSemCom/reproduction/FAJSCCv2_run_check_2026-10-07/upstream/model/JSCC.py:142)；[FADecoder](D:/PyChram_project/CognitiveSemCom/reproduction/FAJSCCv2_run_check_2026-10-07/upstream/model/Decoder.py:149)。
- 输入/输出：中间二维特征图及条件特征 → 选择窗口、偏移、通道/空间注意力 → 同尺寸增强特征。具体返回类型随训练/评估模式不同。
- 真正改变：哪些窗口走较贵的注意力计算。它首先是**算力分配**，不能自动解释为少传了符号。
- 已有状态：最小训练/推理通过，正式性能未复现。
- 借用方式：先保留原 backbone 和执行分支，仅研究 decoder 的选择准则。
- 最小对照：原始路由、同窗口数随机路由、同容量新路由；统一算力/时延与训练预算。
- 需要量化：保留窗口数、排序/索引代价、batch=1 时延、每图质量变化、路由在不同噪声下的稳定性。
- 静态问题：评估中 mask 来自 Gumbel 采样；窗口数量使用跨 batch 均值。不能把一次前向成功当作确定性/逐图预算已验证。

**M2｜PADC / OraNet：低成本质量预测，P0**

- 来源：[PADC 作者仓库](https://github.com/wyzhang-ustb/Predictive-and-Adaptive-Deep-Coding-for-Wireless-Image-Transmission-in-Semantic-Communication)，TWC 2023。
- 核心代码：[OracleNet](D:/PyChram_project/CognitiveSemCom/reproduction/padc/original/OracleNet.py:29)；[原始 codec](D:/PyChram_project/CognitiveSemCom/reproduction/padc/original/models.py)。
- 作者输入：编码特征的每通道均值和标准差、SNR、码率对应的输出通道数；输出一个 PSNR 预测值。
- 已有状态：43,021 参数的 Fading OraNet 检查点严格加载，形状检查通过；当前新机制的预测校准未验证。
- 借用方式：保留“小特征头预测下游效果”的范式。若改成接收端计算收益预测，需要重新定义可部署特征、标签并训练。
- 不能照搬：作者权重不是重传收益、失败概率或适配后 codec 的现成预测器；网络名含 Oracle 不代表它读取未来真值。
- 最小对照：常数、SNR-only、同容量质量头、同容量收益头。标签变化必须带来超出输入/容量差异的增益。
- 已知教训：D56 的三个可观察标量上，收益头没有稳定胜过普通质量头；新方向应检验局部潜变量信息是否带来新增价值。

**M3｜DCAT：动态信道调制与重排，P0**

- 来源：[DCAT 官方仓库](https://github.com/PKU-PCNI/SoM-DCAT)。
- 核心代码：[EDcoder_Ours_Update / My_Scoring / Pred_NMSE_CSI](D:/PyChram_project/DCAT-SoM/SoM-DCAT-main/net/som_modules.py:63)、[Scoring_Channel](D:/PyChram_project/DCAT-SoM/SoM-DCAT-main/net/Scoring_channel.py:47)。
- 数据流：CSI、年龄、速度、SNR → 误差估计与条件调制；信道打分 → 发射次序 → 信道/均衡 → 反向次序恢复。
- 已有状态：公开权重推理矩阵及本地批处理适配已运行审计。
- 借用方式：作为可观察可靠性与时变信道的参考接口；不把真实 CSI 作为新方法部署输入。
- 最小对照：固定同一个权重族与 CSI 权限，比较 hold/预测及匹配适配；不能跨 with_CP/without_CP 权重直接归因。
- 已知教训：其原有模块已包含 NMSE 感知；“加入不确定性”本身不是新的功能描述。
- 具体缺口：原官方网络的梯度/排序设计与本地推理适配并不自动构成完整训练流水线；要训练新层时需独立核查梯度和冻结范围。

**M4｜本地因果 CSI 与反事实评测工具，P0**

- 代码：[csi_predictors.py](D:/PyChram_project/CognitiveSemCom/research/dcat_csi_prediction_20261004/csi_predictors.py)、[test_causality.py](D:/PyChram_project/CognitiveSemCom/research/dcat_csi_prediction_20261004/test_causality.py)、[run_e2e.py](D:/PyChram_project/CognitiveSemCom/research/dcat_csi_prediction_20261004/run_e2e.py)。
- 代码：[suite_codec.py](D:/PyChram_project/CognitiveSemCom/research/jev_image_transmission_20260926/suite_codec.py)、[suite_learning.py](D:/PyChram_project/CognitiveSemCom/research/jev_image_transmission_20260926/suite_learning.py)。
- 可复用：时间因果边界、统一动作结果表、配对噪声、输入白名单、训练标签隔离、资源与等待成本。
- 已有状态：部分组件有实际小试，整套流水线正式全矩阵未完成。
- 不宜照搬：当前 crop→放大→独立重编码→固定 alpha 融合没有通过 codec 门。
- 注意：目录内还包含旧 API 组件；本次图像研究只抽取本地数值/评测部分，不调用 LLM/LAM API。

**M5｜NAFNet 残差块：轻量修复单元，P1**

- 来源：[NAFNet 作者代码](https://github.com/megvii-research/NAFNet)、[NAFBlock 源码](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py)。
- 借用对象：残差与门控计算单元，不把整个图像恢复网络和权重直接当作 JSCC decoder。
- 建议适配接口：特征图 F → F+R(F,u)，其中 u 为可部署条件；这是拟议移植接口，不是作者已提供的通信模块。
- 插入候选：均衡并恢复排列后的 latent，或基础图像重建之后。
- 已有状态：本轮只核查公开仓库/源码入口，未本地运行。
- 最小对照：同参数普通残差 CNN、无条件 NAFBlock、条件 NAFBlock，训练预算一致。
- 风险：图像去噪权重与 JSCC 潜变量分布不匹配；无条件后处理就能取得的收益不能归给信道感知。

**M6｜NTSCC / NTSCC+：概率模型与局部码率，P1**

- 来源：[NTSCC](https://github.com/wsxtyrdd/NTSCC_JSAC22)、[NTSCC+](https://github.com/wsxtyrdd/NTSCC_plus)。
- 学习内容：图像潜变量的概率/熵估计如何影响分配的信道符号数。
- 与 M1 的区别：这里主要控制通信资源，M1 主要控制计算资源。
- 建议卡片接口：潜变量及概率参数 → 分组符号数/分配表 → 接收端可解释的表示。实现应遵照选定版本实际接口。
- 已有状态：作者仓库列出模型入口与预训练权重链接，未在本机验证。
- 最小对照：统一分配、原熵分配、失真收益分配；全计侧信息和总符号数。
- 风险：熵高不一定代表重建收益高；反过来“用失真重要性替代熵”也不能直接宣称新颖。
- 工程边界：NTSCC+ 的 NATTEN 依赖有明确版本警告，NTSCC++ 在线部分仍列 TODO；先验证标准 NTSCC，避免同时移植多个版本。

**M7｜HJSCC / DeepJSCC-f：条件增强与融合，P1**

- 来源：[HJSCC](https://github.com/zhang-guangyi/HJSCC)、[HJSCC_feedback](https://github.com/zhang-guangyi/HJSCC/tree/main/HJSCC_feedback)、[DeepJSCC-f](https://github.com/kurka/deepJSCC-feedback)。
- 直接阅读对象：层级潜变量/反馈编码过程；DeepJSCC-f 的 jscc.py 中 OutputsCombiner、DeepJSCCF。
- 学习内容：后续信息如何以既有接收结果为条件；累积观测如何帮助重建。
- 已有状态：核查公开仓库，HJSCC README 提供两类模型的权重链接；没有下载/验证这些权重。
- 最小对照：同总预算单轮、独立双轮、条件双轮；相同反馈权限、总能量与时延。
- 需要先回答：原第二轮是在补充信息，还是重复传另一张低质量重建？训练和融合是否一起优化？
- 反馈边界：发射端使用的接收重建不能免费获得；反馈内容、噪声、延时、符号成本均需明确。

**M8｜SwinJSCC / Dynamic-JSCC：条件调制与速率控制，P2**

- 来源：[SwinJSCC](https://github.com/semcomm/SwinJSCC)、[Dynamic-JSCC](https://github.com/mingyuyng/Dynamic_JSCC)。
- 学习内容：SNR/码率条件映射、软/硬选择、同一网络覆盖不同工作点。
- 插入意义：提供可靠的可调骨干或对照，不把“加入 attention/FiLM/掩码”单独包装成创新。
- 已有状态：SwinJSCC 本轮核查官方入口；Dynamic-JSCC 本地源码存在，原版当前运行未核验。
- 最小对照：固定工作点模型、统一模型、加入新模块的统一模型；保持训练覆盖与调参机会一致。
- 工程边界：SwinJSCC 官方提示 torch>1.12 下带 SA/RA 的结果可能不一致；不能通过去掉核心模块后仍称完整原版复现。

**一次真正有效的 debug 应留下什么**

| 检查点 | 记录内容 | 用途 |
|---|---|---|
| 原图输入 | 分辨率、RGB/范围、裁剪与归一化 | 排除数据口径差异 |
| encoder 输出 | shape、均值/方差、极值、是否含 batch 混合 | 理解表示和样本边界 |
| 掩码/计算路由 | 每图保留数量、索引、条件改变前后差异 | 验证模块实际动作 |
| 实复映射与归一化 | 每图复符号数、每图发射能量 | 防止带宽或功率优势被漏算 |
| 物理信道 | y=hx+n 的维度、实虚噪声方差、可见 CSI 与真实 CSI 的分离 | 确认 SNR 与因果性 |
| 均衡/反排列 | 前后统计、误差结构、反排列正确性 | 定位失真进入点 |
| decoder 中间层 | 每 stage 特征、重计算与轻分支差异 | 检查计算是否有质量价值 |
| 输出与指标 | MSE、逐图 PSNR、统一 LPIPS、实际总时延 | 避免只看平均涨点 |
| 反向传播 | 非零且有限梯度、真正更新的参数 | 防止有模块却没有学习信号 |

DCAT 的 128×128 输入在四次下采样后形成 64 个空间位置；C128 对应 64×128 个实潜变量，即 4,096 个复载荷符号。C64/C192 对应 2,048/6,144 个复载荷符号。这个计数不包含导频或反馈，且不能与其他项目使用的 CBR 分母混用。

FAJSCC 先按模型 stage 数计算输出通道：C=int(6×4^s/rcpp)，所以不要仅凭参数名 rcpp 猜实际符号数，应从每图 encoder 输出数/2 核算。全局展平配对实虚会把 batch 的样本边界引入物理符号解释，应在 B=1、B>1 与换 batchmate 的条件下检查。

**首轮复现与创新筛选的交付定义**

1. 一张图的完整张量流与预算表。
2. 一个可重复的原版质量—计算/通信预算曲线。
3. 一张 oracle/启发式/原路由比较表，证明或否定可利用空间。
4. 新模块与原模块同输入、同容量、同训练预算的比较。
5. 一页结论：通过哪些条件、在哪些条件失败、还能主张什么。

其中 1–3 是进入方法训练前的证据。若第 3 项显示没有空间，应保留负结果并更换问题，不通过改名、挑种子或增加模型容量延长同一个无效假设。
