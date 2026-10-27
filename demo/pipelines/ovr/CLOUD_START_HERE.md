# 云端试跑：OVR-ColdStart + OVR-RL

Mac 主控，用户自行租卡，Windows 备用。程序只准备环境、下载模型、推理和评分，不创建、开机、续费或关闭付费实例。**当前采用队友固定的 100 题；真实运行状态见 `results/reports/cloud_preparation_status.json`。**

## 2026-10-25：新20题预算对照仅完成本地准备

用户已要求准备8k/16k对照代码，尚未要求现在启动GPU。新入口、题单、40次生成的范围和运行窗口见 [CLOUD_BUDGET20.md](CLOUD_BUDGET20.md)。不要使用下方历史100题控制器、历史截止或旧运行命令来启动本轮。

## 2026-10-23 已完成：两个模型各100题

北京时间16:12完成200次正式生成，16:13确认服务均停止、GPU 0%/0 MiB；已提醒用户关卡。实例开关和计费由用户控制。**不要重启控制器或旧运行命令。**

ColdStart 34/100，RL 31/100；截断48/100、70/100。全部100题ID、顺序及图片一致，200条Mac官方重评分与云端一致；原始SHA未变。完整数据包与离线6案例页已备份Mac。结果入口见 `results/reports/cloud_vllm_shared1008192_v1/pilot/交接说明.txt`。

此后全部运行/暂停/预算准备段落均为历史记录，不能据其重复推理。冻结配置PREPARED字段保留以维持运行指纹；完成状态由 `controller_complete.json`、`mac_verification.json` 和 `cloud_preparation_status.json` 记录。后续只做CPU分析、组员交接和海报整合，新增推理需另有明确指令。

## 2026-10-23 授权与运行记录（历史，现已完成）

用户已明确要求尽快跑完题目、不再增加实验。固定五档难度检查已经完成，后台控制器 `ovr.authorized_pair_20261023` 已于北京时间 09:31 开始正式评测，顺序跑 ColdStart、RL 各 100 题；无需再次等待预算确认，不调用旧 `run_pair.sh`。该次授权截止为 **北京时间 2026-10-23 19:20:08**，从授权起任务最多 10 小时，用户提供租价 2.78 元/小时，对应最多 27.8 元的任务运行时间费用。用户自行关闭实例；程序停止服务不会结束平台实例计费。

固定正式协议为 `configs/cloud_vllm_shared1008192.json`：vLLM 0.11.0，BF16，temperature=0.6、top_p=0.95、top_k=40，seed42/请求，输出8192、总上下文16384；原图、原提示词与100个ID保持不变。每题只保存本协议的首次生成，答错、空输出、重复和截断均保留；基础设施中的不确定 HTTP 尝试必须先检查，禁止静默重采样。输出为 `results/{raw,scored,runs,reports}/cloud_vllm_shared1008192_v1/`。

状态文件为云端 `.build/formal-pair-controller-state.json`，本地授权快照见 `results/reports/authorized_shared100_20261023.json`。当前聊天的提醒会跟进完成/异常及关卡通知。下面 10/22 的“暂停”和“等待预算”段落是历史记录，已由此授权取代；原始配置和结果不改写。

三次单 token 对照的原始严格 logprob 控制未通过，保留记录；事后概率质量复核显示白图差异约为同图差异319倍，支持像素影响输出。该复核是工程审查，不改写原门槛，不声称视觉理解或缓存因果已经证明。正式成绩只计算新协议下完整100题，不混入任何诊断或挑选较好答案。

如需人工查看，`.venv-cloud/bin/python -m ovr.vllm_evaluation --check` 只读验证证据。正式控制器已在运行，不要重复手工启动 `--run`。该子集含工程检查用过的7题，需披露开发重叠；同学的 Qwen greedy/4096 结果若不重跑，须注明与本次 OVR 的协议差异。固定seed不保证 online vLLM逐token一致。

## 当前暂停正式评测（2026-10-22，历史）

**不要执行 `run_pair.sh` 或 `--phase pilot`。正式生成仍为 0；下文正式运行命令仅供后续冻结协议后参考。** 官方 vLLM 部署诊断已结束，所有 GPU 服务已停止，已提醒用户关卡。Mac 已核验并保存原始证据包；现在整理、评分和同步无需 GPU。

详情见 `results/reports/vllm_deployment_audit/诊断结论.txt` 与 `conclusion.json`。仅使用相同的两个诊断 ID，不能把它们当作准确率样本。vLLM 默认采样（temperature=0.6、top_p=0.95、top_k=40）下，ColdStart 两题均正常结束，RL 仍有一题在 4096 截断。单题 8192 长度对照虽然自然结束，但轨迹已改变且带猜测表述，不宣称长度修复或可靠视觉推理。原始 frozen config 和所有旧回答不作替换。

vLLM 独立环境 `/root/ovr-course-env/vllm011` 已通过依赖检查、CUDA 扩展导入和真实推理；版本清单及实际 server 命令保存在证据包。固定一次性诊断入口为 `ovr.diagnostic_checkpoint_defaults`、`ovr.diagnostic_vllm`、`ovr.diagnostic_vllm_length`，后两者遇已有输出会拒绝重新生成；这些入口不是正式评测启动命令。原 Transformers 环境继续保留。

## 下一轮排查已准备，尚未运行

`configs/cloud_vllm_preflight8192.json` 固定每档难度一个新 ID；两个模型各 5 题，另有 RL 三次单 token 图像对照。入口 `.venv-cloud/bin/python -m ovr.repetition_preflight --run-bounded-gpu-preflight` **最多 13 个生成请求、40 分钟保护上限，不会启动 100 题**。等待用户确认云机状态与恢复连接，勿在 Mac 执行 GPU 入口。

它会先检查权重头、服务版本和像素敏感性，再显式传入 0.6/0.95/top_k40/8192 参数；不要仅修改旧 JSON，因为原 Transformers 后端会清空采样参数，旧采样诊断另有显式恢复步骤。所有旧配置和输出保持不变。

新评测门槛针对部署和数据完整性，答错、重复和截断都应如实记录，不按分数或要求所有题都 EOS 来挑选结果。约半小时的检查通过、参数与预算确定后即可开始 100 题；细则见 `results/reports/vllm_deployment_audit/下一轮排查与100题启动条件.txt`。

## 当前采用队友的 100 题（2026-10-22）

用户已选择 `workstreams/shared_test100`：100 个固定 ID，原图字节保持不变。两个抽样种子已重现，全部题目/选项/答案/难度/学科与本地官方数据核对一致。旧 20 题方案保留作历史记录，本次正式范围是同题 100 题、两个 OVR checkpoint 共 200 次生成；canary 的 4 次生成另存。

当前云机的原始镜像带 Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128，使用独立环境复用镜像的 CUDA 库。镜像版本满足上述条件时可用 `bash scripts/cloud/setup_image.sh --no-gpu-check`，不重装整个 CUDA 栈；其他机器继续用下文 Python 3.11/cu126 安装方案。两种运行环境都有版本核验，结果记录实际环境，不能混写同一组输出。

本次入口：

```bash
source /etc/network_turbo  # AutoDL 学术资源加速，仅影响当前 shell
export OVR_CLOUD_CONFIG="$PWD/configs/cloud_shared100.json"
export OVR_PUBLIC_MODEL_ENDPOINT=https://huggingface.co
# 使用自己的已获准 HF 账号登录，token 不加入 Git credential
HF_ENDPOINT=https://huggingface.co .venv-cloud/bin/huggingface-cli login
.venv-cloud/bin/python -m ovr.cloud check --gpu --access
bash scripts/cloud/run_pair.sh
```

提示词逐字对齐共享包的 boxed-options-v1；BF16、SDPA、greedy、4096 输出 token、fast image processor、像素范围与 Qwen 参考设置一致。保留每个 checkpoint 的多模态 chat template 与 `<think>` 预填充，并记录模板哈希。保留原图，由处理器做统一缩放，不另行裁图或重编码。

输出单独放到 `results/{raw,scored,runs,reports}/cloud_shared100/`，结果包为 `artifacts/OVR_CLOUD_SHARED100_RESULTS.zip`。`coverage.json` 必须显示每模型 100 个唯一 ID 且 complete=true，才算完成本次子集评测。

共享包没有 Qwen 原始回答，其参考设置明确使用自定义评分；需要队友提供原始输出后，用统一规则重评分再比较准确率。现在不能直接拼接自定义分数与 MathVision 官方评分。总体分数由固定 100 题的全部输出计算，改善、退化和共同失败案例另作定性展示；论文报告数字与本组实测结果分别标注。

2026-10-22 已完成云机 HF 授权核验，ColdStart 已在真实 GPU 上生成两题，但均在 4096 tokens 截断，正式 100 题尚未启动。RL 也已完成两题，均发生截断；具体检查结果以状态报告为准。后续机器仍需分别完成模型条件确认与云机 token 登录。AutoDL 加速说明：https://www.autodl.com/docs/network_turbo/。

## 本次独立诊断（未计入正式准确率）

真实 ColdStart、RL 各两题均在 greedy／4096 tokens 下出现重复并截断，已保留原回答；100 题正式评测尚未启动。四次独立采样诊断现已完成：ColdStart 两题正常结束、答对 1/2；RL 两题仍截断、答对 0/2。仅为接口诊断，不能推断总体准确率。用户选择先做四次独立采样诊断：相同题目、图像、提示词、精度与 token 上限，只改为 `do_sample=true, temperature=0.6, top_p=0.95, top_k=0`，seed 仍为 42。此参数用于排查，不声称是论文评测设置。

```bash
.venv-cloud/bin/python -m ovr.diagnostic_sampling --model coldstart
.venv-cloud/bin/python -m ovr.diagnostic_sampling --model rl
OVR_CLOUD_CONFIG=configs/cloud_diagnostic_sampling.json .venv-cloud/bin/python -m ovr.cloud report --phase canary
```

入口固定每模型前两题，输出隔离在 `results/{raw,runs,scored,reports}/cloud_diag_sampling_20261022/`，没有正式 100 题启动选项。有效采样参数写入会话记录；重复执行只补未完成 ID，不挑选较好的回答替换原结果。检查记录见 `results/reports/cloud_shared100_canary_review.json`，解码对照见 `results/reports/cloud_diagnostic_sampling_comparison.json` 与 `results/reports/cloud_diagnostic_sampling_summary.csv`。官方部署路径核验现已完成，结论见上方；下一步与队友统一正式协议。配置中的状态文字是冻结时记录，当前进度以 preparation status 为准。

## 需要的机器

单张 **L20 / A40 48GB**，或 A100 / A800 80GB；CPU 8 核以上，系统内存 64GB，数据盘至少 100GB，Ubuntu 22.04 x86_64。建议选较新的 PyTorch 镜像和 NVIDIA 驱动。安装脚本使用独立 Python 3.11 环境，不替换镜像的系统 Python。

核心版本：PyTorch 2.8.0 + torchvision 0.23.0（官方 CUDA 12.6 wheel）、Transformers 4.57.1、Accelerate 1.11.0、HF Hub 0.35.3。以上是原 Transformers 路径；本次另建的 vLLM 0.11.0 诊断环境见上方，不混写为同一套环境。无需量化。

安装过程优先使用国内源，失败后对固定版本回退官方源，不修改全局 pip 配置。

若云机下载多个权重分片时出现连接重置，可设置 `OVR_DOWNLOAD_WORKERS=1` 后续传（默认 2，只允许 1 或 2）。此选项仅改变文件下载并发，保留已完成和临时文件，不改变模型 revision、推理参数或官方哈希校验。

版本来源：[PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/#v280)、[Transformers Qwen2.5-VL](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen2_5_vl)。实际依赖和 GPU 信息会写入运行记录。安装和模型下载需要网络；正式生成只读取本地 checkpoint。

## 1. 先确认 RL 访问权限

在 [OVR-RL 页面](https://huggingface.co/Kangheng/OVR-7B-RL) 用自己的 HF 账号接受访问条件并确认获准下载。不要把密码、token 发到聊天或提交仓库。Mac 未登录时的 GatedRepoError 不代表你的账号一定没权限；目标机器登录后会重新核验。

## 2. 把代码放到云机的数据盘

任选一种：

- 从 Mac 把 `artifacts/OVR_CLOUD_STARTER.zip` 传到云机，解压后进入 `OVR_CLOUD_STARTER`。此包约 3.5MB，含本次固定 100 题、原图和历史 20 题方案，不需要在云机登录 GitHub。
- 或在云机通过自己的 GitHub 登录克隆私有仓库：`https://github.com/Wangjjeupu/ovr-course-project.git`。

AutoDL 可把工作目录放在 `/root/autodl-tmp`。使用云机提供的 SSH 地址/端口传输，不要将登录密码写进脚本。

## 3. 安装一次，再启动

在云机项目目录执行：

```bash
bash scripts/cloud/setup.sh
HF_ENDPOINT=https://huggingface.co .venv-cloud/bin/huggingface-cli login
```

HF 登录时选择**不添加 Git credential**即可。若先用无卡模式准备环境，可运行 `bash scripts/cloud/setup.sh --no-gpu-check`；安装和下载本身不需要 GPU。正式试跑会强制检查 CUDA、BF16 和显存。

建议在 tmux 中启动，避免 Mac 断开 SSH 导致任务终止：

```bash
tmux new -s ovr
bash scripts/cloud/run_pair.sh
```

已有 tmux 会话时用 `tmux attach -t ovr`。运行包不会自动安装 tmux 或控制实例计费。

## 自动流程

1. 核验所选配置的 ID、参考答案分离、图片可读性与 SHA256。本次配置为固定 100 题，难度分布与共享包一致。
2. 对两个固定 revision 的配置和**全部权重文件**做权限 HEAD 检查。任一不可访问就停止，不先下载另一个大模型。
3. 下载两个 checkpoint，每个约 16.6GB；校验上游 LFS SHA256。ColdStart 的 merged 权重通过本地符号链接视图适配标准文件名，原权重及其索引保持原样。
4. 两个模型分别做前 2 题 canary，单独保存。要求真正生成图像张量、回答非空、正常 EOS 结束；有截断就停下检查，不自动改参数或反复抽样。
5. canary 通过后，两个模型**顺序**各跑同一组 100 题（或显式选择的历史配置题目数）。每题落盘并 fsync；重新执行脚本只补齐未完成题，不因答错重跑。
6. 使用仓库内未修改的 MathVision 官方评分器，写入评分副本；生成覆盖报告、准确率及 Wilson 95% CI、分难度/学科结果、长度/耗时、截断和逐题转换表。
7. 本次导出 `artifacts/OVR_CLOUD_SHARED100_RESULTS.zip`（历史 20 题为 `OVR_CLOUD_PILOT20_RESULTS.zip`），含原始回答、评分、图像、协议、环境记录、汇总和 SHA256 清单；不包含模型和认证文件。

## 国内下载源（已配置）

- 普通 Python 依赖：清华 PyPI，`https://pypi.tuna.tsinghua.edu.cn/simple`。[使用说明](https://mirror.tuna.tsinghua.edu.cn/help/pypi/)
- PyTorch / torchvision：上海交大 cu126 wheel 源，`https://mirror.sjtu.edu.cn/pytorch-wheels/cu126`。[使用说明](https://mirrors.sjtug.sjtu.edu.cn/docs/pytorch-wheels)。显式固定 `torch==2.8.0+cu126`、`torchvision==0.23.0+cu126`，避免换成 CPU 或其他 CUDA 构建；失败后回退官方 cu126 源。
- 公开 ColdStart：优先尝试 `https://hf-mirror.com`，调用明确使用 `token=False`，不携带已保存或环境中的 HF token；镜像不可用时回退官方。
- 受限的 RL：权限核验和权重下载始终使用 HF 官方端点；不会把 HF token 发给第三方模型镜像。

镜像不改变模型版本、生成参数或推理精度。除权重的官方 LFS SHA256 外，配置和 tokenizer 文件也会比对官方 Git/LFS 内容哈希。哈希不一致时停止，不继续加载。实际速度和国内源的大文件可达性仍需在目标云机验证；不会宣称镜像一定更快。

想全部使用官方源时：

```bash
OVR_PYPI_INDEX_URL=https://pypi.org/simple OVR_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash scripts/cloud/setup.sh
OVR_PUBLIC_MODEL_ENDPOINT=https://huggingface.co bash scripts/cloud/run_pair.sh
```

Python 3.11 若需由 uv 另行安装，其运行时下载仍走 uv 的原始发布源；优先选择已经带 Python 3.11 的镜像。

## 历史 20 题协议和结果位置（本次使用上方 100 题入口）

`configs/cloud_pilot20.json`：BF16、SDPA、无量化、batch 1、greedy、seed 42、最多 4096 输出 token、8192 总上下文预算。统一图片文件，处理器 min_pixels=200704、max_pixels=1003520，使用每个 checkpoint 自带的**多模态 processor 模板**，保留官方 `<think>` 预填充。

每题 600 秒停止条件会记录为 `time_limit`，输出上限记为 `length`。正式试跑保留这些结果，不删除。会话默认最多 60 分钟，剩余时间不足一题完整时间窗时退出为未完成，等待续跑；它不是云平台计费或关机上限。

这是我们两个模型的 BF16 工程试跑方案，替代尚未冻结的 4-bit 候选方案。请把此配置交给另一位实验同学对齐，不能混合不同协议的分数。20 题试跑不称作 3040 题全量复现。旧 `ovr.cli run` 入口仍为历史方案；本次只用 `ovr.cloud`。

| 内容 | 路径 |
|---|---|
| 原始回答 | `results/raw/cloud_pilot20/pilot/{coldstart,rl}.jsonl` |
| canary 回答 | `results/raw/cloud_pilot20/canary/` |
| 会话、错误、环境、下载权限记录 | `results/runs/cloud_pilot20/` |
| 官方评分副本 | `results/scored/cloud_pilot20/` |
| 覆盖报告和汇总表 | `results/reports/cloud_pilot20/pilot/` |
| 给组员的结果 ZIP | `artifacts/OVR_CLOUD_PILOT20_RESULTS.zip` |

只在 `coverage.json` 中 `complete=true`、两个模型各 20 个唯一 ID 时，才称作完成本次试跑。canary 的 4 次生成与正式 40 次生成分开，不加入正式准确率。

## 续跑与单步操作

断开后重连到相同目录，再运行 `bash scripts/cloud/run_pair.sh`。已完成结果不会重新生成；运行指纹变化（模型、代码、协议、输入、软件/GPU 环境）会拒绝混写。OS 锁会在进程退出后自动释放。若出现 JSONL 末行损坏，保留文件并检查，不静默删行。

以下单步命令执行前，先选择本次配置；否则 CLI 默认使用历史 20 题配置。

```bash
export OVR_CLOUD_CONFIG="$PWD/configs/cloud_shared100.json"
```

单独重建统计和导出：

```bash
.venv-cloud/bin/python -m ovr.cloud report
.venv-cloud/bin/python -m ovr.cloud export
```

单独检查或运行：

```bash
.venv-cloud/bin/python -m ovr.cloud check --gpu --access
.venv-cloud/bin/python -m ovr.cloud download
.venv-cloud/bin/python -m ovr.cloud run --model coldstart --phase canary
.venv-cloud/bin/python -m ovr.cloud run --model rl --phase canary
.venv-cloud/bin/python -m ovr.cloud run --model coldstart --phase pilot
.venv-cloud/bin/python -m ovr.cloud run --model rl --phase pilot
```

运行完成或出错后，云机仍可能持续计费。由用户设置平台定时关机、下载结果并停止实例；脚本不会处理其他项目的实例。
