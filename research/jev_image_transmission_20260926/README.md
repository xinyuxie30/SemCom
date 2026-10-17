# Jev 图像传输实验

本目录与 CognitiveSemCom 的 KG 代码独立。DCAT 仅作为明确记录的外部只读基线；官方 Jev 通过用户已配置的 TypeSafe 直连接口调用，Visual Jev 在本机运行。

先读：

- [预实验真实结果](PILOT_RESULTS.md)：模型已跑通，原始 Jev 和当前两轮 codec 尚未显示优越性。
- [完整实验协议](FULL_EXPERIMENT_PROTOCOL.md)：正式矩阵、对照、分组、资源/延迟模型及结论边界。
- [最初的文献与研究判断](RESEARCH_DECISION.md)：历史调查，执行状态以本页和最新结果为准。

## 启动完整实验

在 PowerShell 执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File 'D:\PyChram_project\CognitiveSemCom\research\jev_image_transmission_20260926\run_full_experiment.ps1'
```

该命令才会正式运行：公开权重动作结果生成 → 官方 Jev / Visual Jev / Qwen → 27 个小网络训练 → 43 种策略的测试与信道延迟回放 → 分组报告。共有 16,224 个状态、最多 16,224 次官方 API 请求、3 个小网络训练种子。大模型和 DCAT 不微调。粗估耗时十几到二十几小时，实际取决于 API/GPU。

依赖已在本机准备。需要当前进程能读到 `TYPESAFE_API_KEY`；不要将密钥写进脚本或聊天。模型推理只把公开图像接收端生成的描述发送到官方 API，不上传原图或标签。调用计费按用户账号执行。

## 只检查环境，不运行实验

```powershell
& 'D:\PyChram_project\DCAT-SoM\.venv\Scripts\python.exe' 'D:\PyChram_project\CognitiveSemCom\research\jev_image_transmission_20260926\run_suite.py' --mode preflight
```

检查数据/权重/代码哈希、跨集合关联图像、CUDA、剩余空间和密钥是否存在。不会向 API 发请求，也不会生成正式测试结果。

## 开发状态端到端检查

```powershell
& 'D:\PyChram_project\DCAT-SoM\.venv\Scripts\python.exe' 'D:\PyChram_project\CognitiveSemCom\research\jev_image_transmission_20260926\run_suite.py' --mode smoke --live-api
```

仅使用 5 张已经观察过的 Kodak 开发图，12 个状态、每小网络 3 轮。它检查全部阶段能否衔接，不能作为论文结果。

## 断点恢复

保存终端打印的 `RUN_OUTPUT` 绝对路径；用同一代码与配置恢复：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File 'D:\PyChram_project\CognitiveSemCom\research\jev_image_transmission_20260926\run_full_experiment.ps1' -Resume 'D:\PyChram_project\CognitiveSemCom\research\jev_image_transmission_20260926\results\formal_suite_实际时间戳'
```

已保存状态不重复运行。请求发送前先记录 attempt；若进程在发送后保存前中断，该状态标记为结果未知并停止补传，不偷偷重新调用。连续 5 次 API 失败或认证失败会中止阶段；更换代码/权重后必须新建运行，不能混进旧结果。Ctrl+C 可停止，已落盘记录保留。

## 输出

每次独立目录 `results/{smoke|formal}_suite_<UTC>/`：

- `manifest.json`、`config.json`、`data.json`、`source/`：状态、分组、配置、代码快照及哈希。
- `states/`、`receiver_arrays/`：部署时可见的接收信息；`labels/`：离线真值和反事实收益。
- `decisions/`、`api_attempts/`：原始模型输出、token、耗时与失败记录，无密钥。
- `learned/`：27 份小网络权重和训练/验证损失。
- `selections/`、`replay/`：每个测试状态的策略、实际执行时刻和资源。
- `fixed_time.csv`、`measured_delay_replay.csv`、`analysis.json`、`REPORT.md`：平均/尾部质量、配对场景区间、种子差异、资源与时延。

`fixed_time` 是统一执行时刻的选择诊断；`measured_delay_replay` 才将实测模型等待推进到信道中，仍属于有明确时序假设的仿真。能直接运行不代表能得到正结果或 SOTA 结论。

## 本机依赖与数据出处

- DCAT：`D:/PyChram_project/DCAT-SoM/.venv/Scripts/python.exe`；官方代码、权重和 A1/A9 轨迹只读引用。
- Visual Jev：本目录 `visual_env/Scripts/python.exe`，固定版本见 `preflight.json`、`visual_assets_manifest.json`。
- UDIS：外部已准备的 `test128`；先合并共享图像的关联组件，选择 512/128/512 张用于本研究训练/验证/测试，其原始角色仍是 UDIS testing。
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)：下载并固定官方 validation HR 的 100 张作为外部测试，文件哈希见 `div2k_manifest.json`。

正式实验尚未启动。最新小试和就绪核验见 `READINESS.md`。
