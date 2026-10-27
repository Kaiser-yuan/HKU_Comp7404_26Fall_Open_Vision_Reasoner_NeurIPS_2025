# OVR 实验操作与离线复现

本包保存本组 OVR-ColdStart 与 OVR-RL 的实验步骤、运行代码、固定输入、原始回答、官方评分和分析结果。使用公开 checkpoint 进行推理评测，未实施 SFT 或 RL 训练。更新日期为 2026 年 10 月 27 日。

## 结果范围

| 实验 | 输入与方法 | 已完成结果 |
|---|---|---|
| 主实验 | 同一批 100 道 MathVision，两个 OVR 模型各一次生成，8,192 tokens 上限 | ColdStart 34/100，RL 31/100；截断 48/100、70/100 |
| 预算补充 | 固定难度均衡 20 题，每模型每题一条 16k 轨迹，提取同轨迹 8k 前缀 | ColdStart 4/20 → 10/20；RL 6/20 → 8/20；共 40 次生成、80 条评分视图 |
| 长度补充 | 固定 1,000 条教师示范与已有同 20 题学生输出，统一 tokenizer | 平均推理 4,444、8,673、至少 10,969 tokens；RL 有 4 条截断 |

主实验是固定 100 题子集，不是完整 3,040 题复现。预算补充为已观察题目上的探索性分析。教师文本题与学生视觉题不同，长度差异不能单独证明能力迁移。Qwen 和 VL-Rethinker 的模型运行由另一实验成员完成，不属于本包的 OVR 运行记录。

## 文件结构

- `docs/COMMAND_ARCHIVE.md`：数据、环境、模型、推理、评分、分析和导出的操作顺序，区分实际回执与重建说明。
- `docs/OBSERVED_STATUS.md`：完成范围、历史诊断和解释限制。
- `docs/AI_USAGE.md`：代码协助与来源声明。
- `ovr/`、`scripts/`：实际运行及复算代码。默认离线入口不会启动 GPU。
- `configs/`、`workstreams/`：冻结参数、同批 100 题、图片、答案、20 题抽样和教师抽样协议。
- `results/raw/`：未改写原始回答；`scored/` 为评分副本；`runs/` 为会话、HTTP 和流式证据；`reports/` 为统计和核验。
- `logs/`、`provenance/`：从实际回执整理的环境信息、模型与源文件身份、源仓库提交。
- `output/`：正式长度报告和图表；`demo/OVR_SHARED100_CASES.html`：可断网打开的真实案例。
- `SHA256SUMS.json`：全部分发文件的哈希。

模型权重、Python 环境、密码、token、SSH 文件及 11.8GB 教师全数据不在包内。固定的 1,000 条教师原始示范与共同 tokenizer 已包含，可以直接离线复算。

## 本地安装与验收

使用 Python 3.11。离线复算入口在 macOS 验收，保留的运行器使用 POSIX 文件锁；Windows 请使用 WSL2，不复制其他机器的虚拟环境。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-trace-lengths.txt
.venv/bin/python scripts/reproduce_technical_demo.py --verify-only
.venv/bin/python scripts/reproduce_technical_demo.py
```

依赖安装需要网络；依赖就绪后的复算无需网络、模型权重、GPU 或 API key。默认入口重新运行 MathVision 官方评分，复核 200 条主实验评分、80 条预算评分视图和长度统计。它只读取分发记录，在全新的 `results/reproduced/<时间戳>/` 中生成副本，核对逐题评分与主要统计文件的字节一致性，检查原始分发文件未变。

成功标志为 `OFFLINE_REPRODUCTION_VERIFIED`，详细证据位于新目录的 `REPRODUCTION_VERIFICATION.json`。重复执行会创建另一目录，不覆盖既有结果。

## 云端运行材料

具体操作见 `docs/COMMAND_ARCHIVE.md`。本包包含实际推理代码和服务命令回执，不在本次交付中重新启动模型。重新生成需单独准备 Linux NVIDIA GPU、相应模型权限及记录中的环境，并使用独立工作副本；新生成不保证与既有采样输出逐 token 一致。

## 来源

- OVR：https://github.com/Open-Reasoner-Zero/Open-Vision-Reasoner
- MathVision 与官方评分：https://github.com/mathllm/MATH-V
- 教师数据：https://huggingface.co/datasets/Kangheng/OVR-Dataset-ColdStart

各来源的版本、许可证与哈希保存在 `sources/`、`vendor/mathvision/` 和 `workstreams/`。以原始记录及冻结配置为证据，历史配置中的 `PREPARED` 字样不改写；完成情况由实际会话和核验记录确定。
