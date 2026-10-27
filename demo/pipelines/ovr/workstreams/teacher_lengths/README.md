# 老师示范与ColdStart RL推理长度分析

本分析只在Mac统计公开数据及已有回答，不训练模型、不执行推理、不使用GPU或付费裁判。

## 数据与冻结抽样

老师来源：作者发布的 `Kangheng/OVR-Dataset-ColdStart`，固定revision `e377de8ec93f9d1ff48744dcdc8c51c4801c6da2`。原始五个Parquet元数据合计1,637,924行，与网站只显示部分数据的viewer数量不同。本发布版不等同于论文全部训练混合。全局均匀不放回抽1000行，seed=20261026；在查看回答前固定行索引。五片分别抽24、174、408、341、53行。协议与原始API清单、Parquet元数据均保留。不能把各片等量平均冒充全局随机样本。

本轮通过官方HF及hf-mirror.com的可续传HTTP Range下载取得全部五片（11,784,309,108 bytes），逐段验证Content-Range和字节数，整文件SHA与上游公布值一致。按原协议取得1,000/1,000条；平均已记录推理4,444.47 tokens，中位数2,522.5，可分段1000/1000。固定抽样协议与下载前commit eb4949f字节一致。此前HTTP/Xet失败记录保留为历史，不再代表当前状态。

学生来源：已完成的 `cloud_vllm_budget20_prefix_v1`，同20道MathVision、同16k预算，各模型只生成过一次。不是重新进行SFT/RL训练。

## 计数规则

统一使用已与两模型checkpoint核验SHA一致的Qwen tokenizer，SHA256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`；文本重新编码时 `add_special_tokens=False`。

- 总输出：全部已保存回答文本的token数，和推理部分分开报告。
- 推理：只计明确think边界内文本，不计分隔符与最终答案。学生的开始think在已冻结的官方模板中预填，因此允许回答开头至唯一 `</think>`；老师数据没有这个模板约定，不能擅自采用隐含开始边界。
- 学生在长度上限内未闭合think：保存已观测推理长度，标记为右删失下界；不当作完整长度。
- 无法明确分段的老师回答：保留总输出，推理长度为null，并报告缺失数量，不填0、不替换抽样。
- 老师数据没有finish_reason元数据；无法据此确定原始生成是否被截断。

## 使用

统计已有学生回答，不需要老师下载或GPU：

```bash
.venv/bin/python -m ovr.trace_lengths
.venv/bin/python -m unittest discover -s tests -p test_trace_lengths.py -v
```

将五个已下载文件放到 `data/cache/teacher_length_v1/`，运行器校验每片大小、上游SHA、元数据行数，然后以memory_map和64行batch只保留事先选中的1000条统计：

```bash
.venv/bin/python -m ovr.trace_lengths --teacher
```

全部分片的可续传下载入口（仅CPU和网络，不下载模型）：

```bash
.venv/bin/python scripts/download_teacher_resumable.py --workers 6 --chunk-mib 8
```

全数据和缓存位于Git忽略目录。完整交接包包含固定1000条原始示范、40条学生回答及共同tokenizer，可在解压目录运行 `python -m ovr.trace_lengths --from-sample` 直接离线复算，无须再次下载12GB数据。复算与原整文件读取统计逐字节一致。详见 [HANDOFF_LENGTHS.md](HANDOFF_LENGTHS.md)。

## 解释范围

学生同题比较是可比的输出行为观察；老师的文本训练题与学生的视觉测试题不同，只能作描述性对照。平均长度增加不证明正确性、认知能力迁移、真正重新看图或超越老师。共同自然结束的16题另列，明确它是按完成状态选择的子集。
