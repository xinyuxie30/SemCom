# 完整实验就绪报告

## Material Passport

- academic-research-suite / experiment-agent / engineering validation。
- 日期：2026-09-28。
- **工程状态：READY。正式实验状态：NOT STARTED。**
- 已就绪的范围：冻结 DCAT 的 Jev 决策、收益学习与延迟回放实验。不是重新从头训练 DCAT/Visual Jev，也不表示科学优势已得到验证。

## 已核验

| 项目 | 实际检查 |
|---|---|
| 官方 API | TypeSafe 直连，固定 jev-1.13.0；真实请求已执行 |
| 本地视觉模型 | Qwen3-VL-4B 与作者 Visual Jev LoRA 已下载并真实运行 |
| 数据 | 512 训练 + 128 验证 + 512 UDIS 测试 + 100 DIV2K 外部测试；准备后哈希核对通过 |
| 防止关联图像跨集合 | 1,106 个原始 UDIS 图像对经共享精确像素连接为 825 个组件，按组件选择与划分 |
| 正式矩阵 | 16,224 状态；最多同数量官方请求；8,544 个测试状态 |
| 对照训练 | 9 类轻量模型 × 3 种子，共 27 个训练任务；基础模型冻结 |
| 单元检查 | 8 项通过，覆盖预算、年龄消融、数据分组、共同噪声、导频刷新、标签隔离及 API 并列/舍入处理 |
| 完整开发小试 | 12 状态，27 个模型，6 个测试状态 × 43 策略 = 258 次回放，五阶段全部完成 |
| 恢复检查 | 同一小试使用 --resume，再次完成且 API attempts 保持 12，未重发 |
| 预算检查 | 最新小试每条回放不超过 6,656 符号，上限实际用到 6,464 |
| 超时路径 | 2 次真实网络失败被保留；测试状态中 20 条相关回放触发 decision_deadline 并正确回退 |
| 正式数据隔离 | 未生成 formal_suite 目录，未运行正式测试推理或正式训练 |

最新完整小试：`results/smoke_suite_20260928T010306111630Z/`。

- `manifest.json`：COMPLETED；states/labels/decisions 各 12，selections/replay 各 6。
- 首次完成五阶段约 95 秒（含阶段启动，不含启动前完整资产哈希检查）；此小试训练只有 3 轮，不能直接线性外推正式训练耗时。
- 12 个官方请求中 10 个返回可用响应，2 个为 URLError；没有删除失败数据。传输可靠性与服务可靠性分开记录。
- 旧小试另有一次 AssertionError；当时未保存失败响应，不能追溯具体字段。新校验器修正了并列选择风险、明确舍入容差，并保存结构化异常响应；离线用例覆盖这些情况，没有将旧异常猜测成确定原因。
- 环境完整包版本见 `environment_versions.json`；实际启动时再次验证 CUDA、文件与密钥存在性。检查不保证未来 API 服务持续可用。

## 正式启动

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File 'D:\PyChram_project\CognitiveSemCom\research\jev_image_transmission_20260926\run_full_experiment.ps1'
```

这一命令会开始正式实验并调用付费 API。此前只完成开发小试与准备。无需补代码、重新下载模型或手工拼接各阶段；失败后保留输出，恢复命令见 README。

正式一次运行的主流程：生成动作结果 → 提取官方/视觉决策特征 → 三种子训练 → 固定时刻选择评估与实测等待信道回放 → CSV/JSON/Markdown 报告和配对区间。估计为十几到二十几小时，接口抖动和回放分支数会改变耗时。

## 科学结论仍待验证

当前能确认“Jev 可以接入并进入实验”，不能确认“加入 Jev 有独立收益”。开发阶段原始 Jev 几乎只选择全局增强，当前两轮 codec 的 oracle 仍低于单轮 C192。完整实验用于检验收益学习是否补足这一点，允许输出否定结果。

此外，ROI 反馈直接前作仍缺全文机制核对；共同预算是上限而非精确等消耗；延迟回放有保守的计算计时假设；计算/服务能耗未测量。详见 FULL_EXPERIMENT_PROTOCOL.md，不能将此就绪报告替代新颖性证明或 SOTA 性能结果。
