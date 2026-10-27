# OVR 实验操作记录

本文件整理已经实施的工作流程与可复用命令，不是完整终端 history。实际执行身份、服务参数、时间和输出以随包 JSON 回执为准。标注为重建说明的命令不作为已执行证据。默认演示只使用 CPU 离线复算。

## 1 数据准备

100 题沿用实验成员提供的固定 MathVision 题单、原图和答案。实际导入实现为 `ovr/shared100.py`，输出为 `workstreams/shared_test100/`；包内已经包含导入后的文件，不需要重新抽题。`validation.json`、`question_ids.json`、`inputs.jsonl` 和 `references.jsonl` 保存身份与顺序。

可复用的输入检查，不生成回答：

```bash
python -m ovr.budget_compare --check
```

该命令的 `CPU_PREPARATION_VERIFIED_NOT_GPU_VALIDATED` 是检查入口的固定状态标签，不覆盖本包中已经完成的云端运行证据。主实验含 7 道工程检查重叠题，名单见主实验核验；20 题抽样排除这些题，使用固定 seed=20261025、每难度 4 题。

## 2 云端环境与模型

实际服务环境为 Linux x86_64、Python 3.12、PyTorch 2.8.0+cu128、Transformers 4.57.1、vLLM 0.11.0。硬件回执报告 NVIDIA GeForce RTX 4090 D、约 47.4 GiB 可见显存；不将该名称推断为标准零售卡配置。`logs/environment_summary.json` 汇集实际会话元数据。

历史项目路径为 `/root/autodl-tmp/OVR_CLOUD_STARTER`；控制 Python 为 `.venv-cloud/bin/python`，vLLM 服务 Python 为 `/root/ovr-course-env/vllm011/bin/python`。当前 Mac 离线分析环境与云端环境分开。

模型固定为：

| 模型 | revision |
|---|---|
| Kangheng/OVR-7B-ColdStart | 52ad9cb58c52d8e5f8c18abee99f77f78e9f6be3 |
| Kangheng/OVR-7B-RL | 53d5680f59196df1533ca5ec55d9046507c1e6d7 |

在新的云端工作副本中，下面是权限检查与权重下载接口的使用方式；不是离线演示步骤：

```bash
export OVR_CLOUD_CONFIG="$PWD/configs/cloud_vllm_shared1008192.json"
python -m ovr.cloud check --access
python -m ovr.cloud download
```

RL 权限由使用者账号取得，凭据通过其自身环境提供，不随包分发。公开 ColdStart 支持匿名镜像并核对上游身份；gated 模型通过官方端点下载。环境历史安装脚本 `scripts/cloud/setup*.sh` 包含早期 Transformers 配置；vLLM 解析快照 `requirements-vllm-audit-lock.txt` 不是全新机器的一键安装承诺，不应直接覆盖现有 CUDA 环境。

## 3 100 题主实验

冻结协议为 `configs/cloud_vllm_shared1008192.json`。两模型使用各自官方模板，同图同题，BF16，采样 temperature=0.6、top_p=0.95、top_k=40，seed=42/请求，输出 8,192、上下文 16,384，batch=1。没有因答错、重复或截断重采样。

实际运行入口是 `ovr.vllm_evaluation.run(model, minutes)`，控制器顺序运行两模型。接口形式如下，其中 `N` 必须替换为独立确认的分钟上限；此处不声称原 shell history 中使用过该字面命令：

```text
python -m ovr.vllm_evaluation --run --model coldstart --minutes N
python -m ovr.vllm_evaluation --run --model rl --minutes N
```

工程诊断入口保存在代码中，已完成的诊断不属于新的运行步骤。正式运行的前置证据保存在 `cloud_vllm_preflight8192_probability_review_v1`；早期重复输出并未被解释为已经彻底消除。

实际服务命令保存在 `results/runs/cloud_vllm_shared1008192_v1/pilot/*/backend-*/*-server.json`，各题提交/回执与会话身份保存在同目录 `http-events-*.jsonl` 和 `session-*.json`。

## 4 20 题 8k 与 16k 对照

准备代码：`ovr/budget_compare.py`；固定题单：`workstreams/budget20/selection.json`；协议：`configs/cloud_vllm_budget20.json`。

实际启动回执记录的命令：

```bash
cd /root/autodl-tmp/OVR_CLOUD_STARTER
.venv-cloud/bin/python -m ovr.budget_compare --run --minutes 180
```

这条命令已经执行并完成，保存在 `results/runs/cloud_vllm_budget20_prefix_v1/launch-receipt.json`。不是本包的默认复算命令。40 条真实轨迹分别给出同轨迹 8,192 与 16,384 token 视图，共 80 条评分。未独立请求第二条 8k 回答。`stream-*.jsonl` 保存原始 SSE 证据，逐题 raw 记录引用 journal 哈希。

运行结束后，CPU 评分与导出接口为：

```bash
python -m ovr.budget_compare --report
python -m ovr.budget_compare --export
```

这些直接接口写入指定命名空间；归档演示使用第 6 节隔离入口，保留随包原统计不变。停止模型服务与平台实例关机分开记录，实例计费由使用者控制。

## 5 教师与学生推理长度

教师源为 `Kangheng/OVR-Dataset-ColdStart`，revision `e377de8ec93f9d1ff48744dcdc8c51c4801c6da2`。在查看回答前按五个分片的全局行号固定 1,000 行，seed=20261026。原始分片共 1,637,924 行、11,784,309,108 bytes，五片上游 SHA 全部核验。

本轮实际使用的下载与原始分片统计接口：

```bash
python scripts/download_teacher_resumable.py --workers 12 --chunk-mib 8
python -m ovr.trace_lengths --teacher
```

采用可续传 HTTP Range，使用官方与匿名镜像端点。既有失败记录仍保存在长度报告目录。全量教师分片未装入本包；1,000 条固定样本已经包含，无需重复下载。独立样本复算接口为：

```bash
python -m ovr.trace_lengths --from-sample
python scripts/build_teacher_length_figure.py
```

统一 Qwen tokenizer，排除 think 分隔符与最终答案；学生开始 think 由官方模板预填。RL 的未闭合截断长度按右删失下界保留。教师原始生成终止状态未知。分片原始统计与样本复算的文件字节一致。

## 6 隔离离线复算

以下命令是本包新增的统一演示入口，不是原始推理命令：

```bash
python scripts/reproduce_technical_demo.py --verify-only
python scripts/reproduce_technical_demo.py
```

入口先核对全部分发文件，然后在新的 `results/reproduced/` 子目录复制必要证据，重新执行未修改的 MathVision 官方评分和统一 tokenizer 统计，比较逐题评分与主要汇总。原始文件保持不变。该入口不下载数据、不加载模型、不训练、不提交 HTTP 推理请求。

## 7 导出与保存

统一打包实现为 `scripts/package_technical_demo.py`，从明确的代码、固定输入和指定结果目录收集文件，排除权重、环境与凭据，生成成员 SHA 和总包回执。下载与租卡不是打包过程的一部分。默认复算的完成证据存入新建的 `REPRODUCTION_VERIFICATION.json`。
