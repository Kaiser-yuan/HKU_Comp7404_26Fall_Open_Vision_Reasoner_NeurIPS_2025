# OVR 四模型评测与输出预算实验复现包

这个包整合本组 Qwen、VL-Rethinker、OVR-ColdStart、OVR-RL 的实际代码、固定 MathVision 题目、原始回答及补充实验。使用 CPU 离线重算即可复现已有结果。

本组实施的是公开 checkpoint 推理评测与分析，没有重新进行 SFT 或 PPO 训练。包内不含模型权重、虚拟环境或登录凭据。

## 最快开始

解压后进入本目录，在 macOS、Linux 或 Windows WSL2 中运行。路径含空格时请加引号。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-offline.txt
python scripts/reproduce_all.py --verify-only
python scripts/reproduce_all.py
```

也可以用 Python 3.12 替代 3.11。第一次安装依赖需要网络；之后重算不访问网络、不调用模型、无需 GPU。不要把别人机器上的 `.venv` 复制过来。

成功会打印 `ALL_OFFLINE_REPRODUCTION_VERIFIED`。总结果位于新建的 `outputs/<时间戳>/RESULTS.md` 和 `VERIFICATION.json`；OVR 原有独立复算同时写入 `pipelines/ovr/results/reproduced/<时间戳>/`。每次使用新目录，原始数据不变。校验模式只检查文件和数据，不会执行官方重评分。

## 每一步做什么 会得到什么

|步骤|作用|完成后可检查的结果|
|---|---|---|
|1 核验文件和数据|检查源文件 SHA256；核对四模型题目、顺序、图片、提示词和答案；核对 Qwen 20+80 无重复|`INPUT_AND_FILE_CHECKS_PASSED`；完整运行生成 `data_validation.json`|
|2 合并 Qwen 并复算历史分析|按 ID 合并前20题与后80题，重新执行自定义答案匹配和 Qwen/VL 配对分析|Qwen 历史复核28/100，VL32/100；配对16共同正确、16改善、12退化、56共同错误|
|3 复算 OVR 主实验|从原始回答重新执行包内未修改的 MathVision 官方评分器|ColdStart34/100、RL31/100；截断48、70|
|4 复算 Effect of Output Budget|对每题16k轨迹及同轨迹8k前缀分别官方评分|ColdStart4→10/20、截断11→0；RL6→8/20、截断15→4|
|5 复算教师与学生推理长度|用包内共同 tokenizer 对1000条教师示范和40条学生回答重新计数|教师4444.473、ColdStart8672.9、RL10968.75 tokens；RL均值为观测下界|
|6 四模型统一官方重评分|把四模型原始回答交给同一个官方评分器，另存评分，不改历史结果|`official_four_models/metrics.csv` 和100题逐题配对表；新分数不必等于旧的自定义匹配|
|7 核验并汇总|确认重算文件、逐题评分与存档结果一致，确认原分发文件未改动|`ALL_OFFLINE_REPRODUCTION_VERIFIED`、`VERIFICATION.json`|

准确率复现是从**已保存的相同输出**重算。重新在 GPU 生成，即使固定 seed，也不能承诺逐 token 或准确率完全相同。

## 原始结果与评分口径

|实验|模型|样本|历史成绩|截断|
|---|---|---:|---:|---:|
|Qwen/VL 4096 greedy|Qwen2.5-VL-7B-Instruct|100|自定义初始26，复核28|2|
|Qwen/VL 4096 greedy|VL-Rethinker-7B|100|自定义初始30，复核32|0|
|OVR 8192 sampling|OVR-ColdStart|100|官方34|48|
|OVR 8192 sampling|OVR-RL|100|官方31|70|

统一评分只统一了答案提取与判分，不能消除推理参数差异。Qwen/VL为 Transformers、greedy、4096；OVR为 vLLM、采样、8192。四者同题同图，但不能据此建立严格的四模型能力排名或 RL 因果结论。原 Qwen checkpoint 的下载 revision 未记录，作为已知复现限制保留。

本包交付前实际统一官方重评分为：Qwen28/100、VL32/100、ColdStart34/100、RL31/100。这里Qwen/VL的总数恰好与旧复核相同，但逐题提取规则仍不同，不代表两种评分代码等价。具体逐题输出与验收回执见 `reference_results/`。

## 输出预算结果与结论

|模型|输出预算|官方判对|长度上限命中|平均输出token|
|---|---:|---:|---:|---:|
|ColdStart|8192|4/20 20%|11/20 55%|6742.1|
|ColdStart|16384|10/20 50%|0/20 0%|8964.15|
|RL|8192|6/20 30%|15/20 75%|7338.1|
|RL|16384|8/20 40%|4/20 20%|11214.2|

每模型每题只生成一次最多16384 tokens的回答，8192视图是同轨迹的token前缀：**40次真实生成、80条评分视图**。不是80次独立生成，也不是旧100题回答的延续。20题来自已经评过的100题，每难度4题，排除7道工程开发重叠题；属于探索性补充，不替代主实验。

结论应表述为：**评测表现受到输出预算、答案提取与答案收束影响；增加预算帮助部分回答完成并得到正确答案，但RL在16k仍有4题截断，更多推理也不保证正确。目前不能据此认定RL本身导致能力下降。**

自然结束、出现完整 boxed 答案、官方评分正确是不同指标。RL的8k有2题还没有完整boxed，却因正文候选被官方评分器判对。RL ID598从8k的评分正确候选变成16k错误最终答案，不能写成“已完成的正确答案被推翻”。

## 包内文件

- `pipelines/qwen_vl/`：你的原技术演示项目，包括20+80原始Qwen结果、VL100题结果、实际执行脚本、环境记录、案例。
- `pipelines/ovr/`：同学ZIP内项目，含OVR100题、预算20题、教师1000条固定原文、tokenizer、原始流式token证据、官方评分器与原测试。
- `sources/user_project_scripts/`：你项目顶层脚本的历史副本，用于核对版本，不与正式入口混用。
- `sources/team_reports/`：两份同学Word原件，未修改。
- `scripts/`：统一CPU离线复算入口。
- `docs/`：逐步说明、实验协议、结果解读。
- `reference_results/`：交付前实际离线重算的汇总与验证回执。
- `provenance/source_inventory.json`：输入ZIP哈希、每个原件哈希及来源。
- `MANIFEST.sha256.json`：本包分发文件校验清单。

首次阅读顺序：本README → `docs/01_逐步复现与预期结果.md` → `docs/03_实验结果与结论边界.md`。想看真实问答案例，可打开 `pipelines/ovr/demo/OVR_SHARED100_CASES.html`，或 `pipelines/qwen_vl/results/model_comparison100/CASEBOOK.md`。

这次整合保留两个旧项目的代码与证据，不覆盖原文件；同学项目中的历史云端地址、截止时间和授权记录是历史资料。新的运行统一从本README和新入口开始。

代码与分析使用了Codex辅助，真实模型生成由本组执行。数据、图片和上游评分器的来源及许可随各原项目保留，见 `docs/04_来源与版本.md`。
