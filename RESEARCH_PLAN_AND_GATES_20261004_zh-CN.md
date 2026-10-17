# 研究规划与门槛：Jev ROI 线收口诊断 + DCAT CSI 预测主攻（2026-10-04）

## Material Passport

- 类型：研究规划（plan + gates），不是实验结果，不构成新颖性证明。
- 背景：2026-10-04 评审结论——Jev ROI 线不按现状跑全矩阵（上界论证）；本规划给出
  收口诊断、限时 codec 门槛、以及 DCAT CSI-prediction 主攻的步骤、kill criteria
  与资源预算。
- 关联材料：`research/jev_image_transmission_20260926/`（Jev 线，READY 未启动）、
  `D:/PyChram_project/DCAT-SoM/`（只读基线 + 本地复现层）。
- 本规划允许全部阶段输出否定结果；每个阶段先做最便宜的决定性诊断。

---

## 0. 决策背景（为什么不是"直接跑全矩阵"）

- **上界论证**：任何 selector 的期望 PSNR ≤ 逐状态 oracle。P1 已测出
  C128+C64 oracle = 24.706 dB < 等载荷单轮 C192 = 25.602 dB（−0.896 dB）。
  换 selector、加 Jev、训练更久都无法翻越——这是 codec/动作集/融合规则的结构性质。
- **P2 补充证据**：同一 32 状态集合内，"保留首轮"（24.562 dB）> "固定全局增强"
  （24.295）≈ 官方 Jev（24.328）——当前 codec 下"不增强"是最优部署策略。
- **Jev 无选区行为**：官方成功响应 31/31 选全局，命中率 34.4% 与 fixed-global 相同；
  Visual Jev 从不选象限。
- 因此：先花 1 小时零成本诊断决定 ROI 动作空间是否存活；codec 修复限时 3 天；
  主资源转到 DCAT CSI-prediction（有实测 headroom、明确基线、零 API 成本）。

---

## 1. Phase 0：Jev 线收口诊断（约 1 小时，零 GPU、零 API）

输入（全部已存在，2026-10-04 已核对结构）：

- `research/jev_image_transmission_20260926/results/counterfactual_20260926T112638517856Z/action_outcomes.jsonl`
  （4,608 行 = 384 状态 × 6 动作 × 2 个 alpha，含逐行 `mse`/`psnr_db`/`single_mse`）
- 同目录 `receiver_states.jsonl`（384 状态）、`summary.json`（聚合）。

新建脚本 `analyze_phase0.py`（放 Jev 线目录），产出 `results/phase0_diagnostic_2026XXXX/`，
做三件事（主分析 α=0.5）：

1. **oracle 动作分布**：逐状态取 MSE 最小动作，统计四象限 / 全局 / 停止的计数。
2. **收益分解**：oracle 相对 best-fixed 的 +0.314 dB 中，"停止 vs 增强"与"选对区域"
   各贡献多少（分别在 {象限∪全局, 停止} 与 {四象限} 子集上重算 oracle）。
3. **全局 vs 最佳象限**：全局增强在多少比例状态上不劣于最佳象限（容差 1%）。

**Kill criteria（任一成立 → ROI 象限叙事死亡）**：
- 最佳象限占比 < 5%；或
- 全局增强在 ≥90% 状态上不劣于最佳象限。

→ 结论写回 Jev 线文档；Track A 降为可选；直接进 Track B。
若象限有意义 → 执行 Track A。

---

## 2. Track A：codec 门槛实验（限时 3 天，RTX 5060 Ti）

目标：第二轮改为**信息可加**，使 oracle 过原始门槛（与 P1 同一门槛、同一 384 状态
协议，结果可直接对比）：**oracle ≥ 等载荷单轮 C192 且对 best-fixed ≥5% MSE 改善**。

变体（成本递增，过即止）：

- **A1（≤0.5 天）learnable fusion**：保留现有 crop-upscale-C64 路径，把固定 α=0.5
  融合换成小 CNN（输入 x_hat_base 局部窗口 + x_hat_enh，输出融合图）。
- **A2（1–2 天）conditional refinement decoder**：冻结 DCAT enc1/dec1；新训
  enc2/dec2（C64）：dec2 以 (x_hat_base, y2) 为输入；象限动作用 region indicator
  条件化；信道在环训练（复用 `suite_codec.py` 已审计的噪声/时间索引实现）。

协议：训练用 UDIS 训练划分（512 图，`suite_data.json` 已冻结），评估用与 P1 完全
相同的 Kodak 384 状态（2 SNR × 2 年龄 × 3 轨迹 × C128 基础）；重算
oracle / best-fixed / C192 三列；α 由模型学，不再固定。

Gate 分支：
- 过 → 重看 oracle 动作分布（象限是否被选中）→ 进 §5 Phase 2（缩减版决策实验）。
- 不过（两个变体都试完）→ ROI 线正式收口归档；全部资源转 Track B。

---

## 3. Track B：DCAT CSI prediction（主攻，2–3 周）

### 3.1 已核实的接口与数字（2026-10-04）

- DCAT 的 "CP"（channel prediction）= `SoM-DCAT-main/utils.py::CSI_casual`：
  **零阶保持**（把最近导频的 CSI 保持到本段末尾）。`with_CP` 权重消费该输出；
  `without_CP` 消费 `NoPred` 老化 CSI。`inference.py` 中
  `CSI_KNOWN = A['CSI'] if with_channel_prediction else A['NoPred']`。
- **理想预测 headroom**（C128，9 个 SNR 平均，2,212 图，本地复现报告
  `results/20260920_232752_published_inference_suite/REPORT.md`）：速度
  2/6/10/15/21 m/s → **+0.55 / +0.68 / +0.80 / +1.35 / +2.20 dB**。
- **零阶保持 NMSE**（DCAT Table II，本地 `channel_protocol_audit.json` 已核对）：
  A1..A9 = **0.034 / 0.092 / 0.157 / 0.471 / 1.134**（A9 的 hold 比零信道先验还差）。
- 09-28 pilot（**未固化，仅会话记录，仓库无脚本**）：Wiener 预测在 A1/A3 闭环、
  A5 减半、A7/A9 失败（NMSE ~0.6–1.1）。B0 第一件事是重建为可复现脚本。
- 评测入口：`reproduction/run_published_inference_suite.py`（--resume，batch adapter
  已审计）；信道轨迹 `SoM-DCAT-main/Channel_inputs/A*.mat`
  （字段：`CSI` / `NoPred` / `AgingTime` / `Doppler`）。
- 研究问题：把零阶保持替换为更好的**因果**预测器，能在多大程度上回收理想预测
  headroom 并转化为端到端 PSNR？（零额外导频开销——只用既有历史导频。）

### 3.2 步骤与门槛

- **B0（1–2 天）固化 + 可达性诊断 + lit check**
  - 重建 09-28 pilot 为脚本：hold vs Wiener vs ideal 的 NMSE（五速度），哈希入档。
  - **genie-linear 可达性诊断**：用轨迹真实自相关计算最优线性预测（genie Wiener）
    NMSE；四线对比 hold / Wiener(估计相关) / genie-linear / ideal，判断每个速度上
    "可达空间"有多大。若 A9 的 genie-linear ≈ hold，则高 Doppler 是信息论限制，
    预测无法回收该处 headroom，叙事限缩到可达速度段（并如实报告 A9 边界）。
  - lit check：CSI prediction for JSCC / semantic communication（paced urllib +
    OpenAlex/arXiv，遵守本机网络限制）；确认"预测器替换 hold + 端到端增益"未被发表。
  - **Gate B0**：headroom 复现一致（±0.1 dB）且存在可达空间且无撞车。
- **B1（3–5 天）预测器阶梯**
  - baseline：hold（现状）、AR(1)、Wiener（估计相关）。
  - learned：GRU/TCN（历史导频窗口，跨速度联合训练；处理非平稳——Case B 速度时变）。
  - 指标：NMSE vs 预测 horizon × 速度（A1..A9）。
  - **Gate B1**：A5/A7 相对 hold 显著改善（如 NMSE 相对下降 ≥40%），A1/A3 不回退；
    A9 给出与 genie 界一致的解释。
- **B2（3–5 天）集成与端到端**
  - harness 扩展：允许 CSI override（绕过零阶保持，直接喂逐符号预测）；作者代码
    不动，扩展写在 `reproduction/` 层并文档化。
  - 用 `with_CP_C128` 模型；UDIS 子集（~500 图）× A1/A5/A7/A9 × SNR {0,6,12}。
  - **Gate B2**：A7 端到端 ≥ +0.3 dB over DCAT-with-CP 基线（同样例、同预算、零额外
    导频开销）；A1 无回退（≤0.05 dB）。
- **B3（3–5 天）完整评估 + 写作**
  - 2,212 图 × 5 速度 × 9 SNR（复用 --resume 入口）；消融：预测器类型 / 历史窗口 /
    horizon / 速度泛化（训练低速 → 测试高速）；对照：hold、Wiener、ideal。
  - 论文定位：对已复现 SOTA（DCAT）自身 CP 模块的改进——"零阶保持 → 因果预测"，
    端到端验证；与"复现 SOTA 再改进"路线一致。

---

## 4. 门槛汇总

| 阶段 | 预算 | 通过条件 | 失败动作 |
|---|---|---|---|
| Phase 0 | 1 小时 | 最佳象限占比 ≥5% 且全局不主导 | ROI 象限叙事死亡；Track A 可选化 |
| Track A | ≤3 天 | oracle ≥ C192 且 ≥5% MSE over best-fixed | ROI 线归档，全转 B |
| B0 | 1–2 天 | headroom 复现 + 存在可达空间 + 无撞车 | 限缩问题范围或转兜底 |
| B1 | 3–5 天 | A5/A7 NMSE 相对 hold ↓≥40%，低端不回退 | 检查 genie 界；限缩叙事 |
| B2 | 3–5 天 | A7 ≥ +0.3 dB，A1 不回退 | 降级为预测器 benchmark / 兜底 |
| B3 | 3–5 天 | — | — |

并行建议：Phase 0 与 B0 的 lit check 可在同一天并行（都不占 GPU）。
总时间：Phase 0 + Track A + B ≈ 2.5–4 周（不含写作）。

---

## 5. Phase 2（仅在 Track A 通过后）：缩减版决策实验

- 1/4 状态子集（~4k 状态）+ 官方 Jev API 抽样（~1k 请求）。
- 主比较：收益学习器（数值+描述 vs +Jev）；消融：年龄 / 停止。
- 先看缩减版结果，再决定是否跑全矩阵（代码已就绪，成本约 1 天 + ≤16k API 调用）。

---

## 6. 资源、成本、边界

- GPU：RTX 5060 Ti 16GB 足够（Track A 的 C64 级小模型、B1 的 GRU/TCN 都远小于
  现有视觉分支）。
- API：Track B **零 API 成本**（相对 Track A/Phase 2 的关键优势）。
- 目录：Track B 新代码放 `D:/PyChram_project/DCAT-SoM/csi_prediction/`（新目录）或
  `reproduction/` 扩展；`SoM-DCAT-main/` 作者源码与官方权重保持原样（只读）。

## 7. 明确不做

- 不按现状跑 16k 全矩阵（已知负结果；代码保留，选项不关闭）。
- 不补 Laya（不解决上界问题）。
- 不实现"执行时重估"（依赖 Track A 通过；且在 2.5 s deadline + ~1.9 s API 等待下
  退化）。
- 不改 DCAT 作者源码（harness 扩展限于 `reproduction/` 层并文档化）。

## 8. 兜底

- Track B 若在 B0/B1/B2 任一 gate 失败：转"CSI 预测在动态 JSCC 中的可达性边界"
  评测型论文（把 genie 界与实测负结果写清楚），或完成 P3R D42 write-up（次选）。
- 所有阶段结果（含负）写回本仓库 dated 文档，保留 kill 记录。
