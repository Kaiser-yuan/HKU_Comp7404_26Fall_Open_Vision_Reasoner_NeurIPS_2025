# OVR 教师示范与学生推理长度完整对比

本分析比较作者公开冷启动数据中固定随机抽取的1,000条教师示范，以及同20道MathVision视觉题上的ColdStart与RL输出。统一tokenizer后，平均推理长度分别为4,444、8,673与至少10,969 tokens。RL有4条截断；跨任务三方统计作为描述性参考，学生同题结果用于比较输出长度。

| 对象 | 样本 | 平均推理tokens | 推理中位数 | 平均总输出tokens | 已知截断 |
|---|---:|---:|---:|---:|---|
| 教师文本示范 | 1000 | 4,444 | 2,522.5 | 4,941 | 未知 |
| ColdStart视觉 | 20 | 8,673 | 9,624.5 | 8,964 | 0/20 |
| RL视觉 | 20 | 10,969 下界 | 11,508.5 | 11,214 | 4/20 |

学生同题观察：RL的平均已观测推理比ColdStart多26.5%，在17/20题更长。两者都闭合推理的16题中，均值分别8,011与9,615 tokens，RL约长20.0%；该16题子集按完成情况筛选，存在选择偏差。

教师推理可明确分段1000/1000条；未分段0条。抽中行对应1000个不同题目文本哈希和1000个不同回答哈希；按原数据行抽样，未进行去重。

## 方法与限制

- 教师样本来自固定公开revision e377de8ec93f9d1ff48744dcdc8c51c4801c6da2。五个原始Parquet分片共1,637,924行，合计11.78GB，全部与作者公布SHA256一致。网页viewer只覆盖部分数据，未作为抽样总体。
- 在查看回答前冻结全局均匀不放回1,000行，seed=20261026；五片分别24、174、408、341、53行。不按长短、正确率或反思关键词挑选。这个公开版本不被假设等同于论文全部训练混合。
- 两个学生模型沿用已完成的同20道MathVision题、同16k输出预算，未新增推理或训练。统一用已核验的Qwen tokenizer重编码，推理长度排除think分隔符和最终答案；学生开始think由原官方模板预填。
- 教师数据没有原始finish_reason，无法统计其真实截断率或确认原始生成预算。学生RL的4条未闭合推理保留为右删失下界，不能把它们当完整推理。不同任务、模态和生成条件使教师与学生之间只能作描述性对照。

## English Summary

Using the same Qwen tokenizer, we measured reasoning sections in 1,000 uniformly sampled demonstrations from the released OVR cold-start dataset and student outputs on 20 MathVision questions. Mean recorded reasoning lengths were 4,444, 8,673, and 10,969 tokens for teacher traces, OVR-ColdStart, and OVR-RL, respectively. Student outputs shared a 16,384-token cap; four RL traces were truncated, making its mean a lower bound. RL produced longer observed reasoning on 17 of 20 paired questions. Teacher traces and student outputs differ in tasks and generation conditions, so these are descriptive length statistics, not evidence of stronger reasoning or teacher-to-student capability transfer.

数据来源：https://huggingface.co/datasets/Kangheng/OVR-Dataset-ColdStart

完整数据与样本哈希、逐条计数、复算检查及图表随交接包保存。之前的下载失败与阶段结果保留。全程CPU，无新模型生成、无SFT/RL训练、无付费模型裁判。
