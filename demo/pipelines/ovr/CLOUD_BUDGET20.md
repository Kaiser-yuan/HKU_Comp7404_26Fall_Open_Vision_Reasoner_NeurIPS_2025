# 20题：8k / 16k推理预算对照（2026-10-25）

状态：**2026-10-26已完成40次真实生成和80条官方评分视图，Mac重评分与云端一致。** 用户自行开启、关闭云实例；程序仅停止自己启动的模型服务，停止服务不等于停止平台计费。

生成时间为北京时间09:16至11:19，11:27确认GPU空闲并提醒关卡。结果：ColdStart 8k为4/20、16k为10/20；RL 8k为6/20、16k为8/20。截断分别11→0与15→4。结果与Mac验收见`results/reports/cloud_vllm_budget20_prefix_v1/`，组员Word见`output/OVR_20题8k16k对照结果_组员简版.docx`。以下命令作为协议记录保留，不代表需要重新启动实验。

## 这轮回答什么问题

较大的输出预算，是否让OVR把已有推理整理成正确的最终答案，还是只是继续重复？这是现有100题结果之后的探索性补充，不是独立留出测试，也不重做SFT/RL训练。

- 从队友同一批100题中，按难度每档4题，共20题。seed=20261025；排除此前7道工程开发重叠题。抽题只使用ID和难度，不使用旧输出、正确率、截断或改善/退化标签。
- 两个模型：原来已核验的ColdStart和RL，模型revision、原图、原提示词、官方chat template与采样参数保持不变。BF16、vLLM 0.11.0、temperature=0.6、top_p=0.95、top_k=40、repetition_penalty=1、seed42。
- **40次真实生成，80条评分视图。** 每模型每题只生成一条最多16,384 tokens的轨迹，同时保存前8,192个token的文本。早于8k自然结束时，两种视图相同；不会强迫模型生成满16k。
- 8k是同轨迹的token前缀，**不是**另一次8k请求，也不保证与旧100题中对应回答相同。旧34%/31%和原始数据全部保留。
- 总上下文24,576，显式检查图片/题目输入token数+16,384不超限；每题最多900秒。检查过程不另加生成探针。

题单和规则在`workstreams/budget20/selection.json`，配置在`configs/cloud_vllm_budget20.json`。本组20题难度均衡，不能直接用其总准确率代替原100题的成绩。

## 开卡之前

Mac只用现有本地环境：

```bash
.venv/bin/python -m ovr.budget_compare --check
.venv/bin/python -m unittest discover -s tests -p test_budget_compare.py -v
.venv/bin/python scripts/cloud/build_budget_starter.py
```

新传输包为`artifacts/OVR_BUDGET20_STARTER.zip`，不覆盖历史运行包。包含代码、原100题输入/图片及选中20题清单、官方评分器、配置、文档和SHA清单；没有权重、认证信息或真实新结果。

可在无卡模式传包和核对文件，不需要重新下载模型或重装CUDA。沿用原云端项目`/root/autodl-tmp/OVR_CLOUD_STARTER`、已有`.venv-cloud`和`/root/ovr-course-env/vllm011`；更新包仅包含准备文件，不覆盖任何历史raw/scored/runs结果。优先继续使用此前48GB显存的实例，实际24k上下文的KV缓存容量要到开卡后由vLLM启动验证。不要据本地测试宣称已完成GPU验证。

## 开卡后，由当前聊天确认连接并执行

首次开卡只做必要的CUDA/显存、版本、缓存权重/配置和服务启动检查，然后直接运行上述40个请求。没有新增小样本诊断、没有自动100题或32k流程。

```bash
cd /root/autodl-tmp/OVR_CLOUD_STARTER
.venv-cloud/bin/python -m ovr.budget_compare --check
# 下行仅在用户已开卡、确认本轮运行窗口后执行。
# 180分钟是示例保护上限，不是速度预测；旧100题授权/截止不沿用。
.venv-cloud/bin/python -m ovr.budget_compare --run --minutes 180
```

请放在tmux中运行。运行器顺序加载两个模型、batch=1，不申请实例、不续费、不改变计费。启动失败、显存不足或网络/生成异常时保留证据并停止，不自动改参数。剩余时间不足一题900秒窗口及清理余量时，保存已完成题并退出，状态明确标记未完成。

每次提交前落盘HTTP事件；流式token、文本、时间保存在每题独立journal。提交后没有完整可信回执的题视为不确定，**不会重采样**。仅对从未提交过的题允许后续显式续跑。答错、空输出、截断和重复全部保留。

## 评分和交接不需要GPU

生成结束后先确认`controller-state.json`的`servers_stopped=true`和GPU空闲，提醒用户可以关卡。评分可在云端CPU完成，也可取回Mac运行；不需要下载模型权重到Mac。

```bash
python -m ovr.budget_compare --report
python -m ovr.budget_compare --export
```

Mac使用`.venv/bin/python`；云端使用`.venv-cloud/bin/python`。评分器为仓库内未修改的MathVision官方代码。原始16k轨迹和流式journal不改写，评分写入独立副本。取回Mac需传输本命名空间的`results/raw`及`results/runs`，不是只传summary。

- 原始轨迹：`results/raw/cloud_vllm_budget20_prefix_v1/{coldstart,rl}.jsonl`
- 流式证据/状态：`results/runs/cloud_vllm_budget20_prefix_v1/`
- 每预算评分副本：`results/scored/cloud_vllm_budget20_prefix_v1/`
- 统计：`results/reports/cloud_vllm_budget20_prefix_v1/{comparison.json,summary.csv,paired.csv}`
- 完整结果包：`artifacts/OVR_BUDGET20_8K_16K_RESULTS.zip`，导出时核验CRC和每个成员SHA256。

每模型需20个唯一ID，两种预算均覆盖全部题，才标记complete。缺失任务和不确定提交单独保留，不用部分结果冒充20题完成。

## 主要比较与下一步

固定报告准确率/95% Wilson区间、8k→16k答对/答错转换、自然结束数、达到上限数、空输出、完整非空boxed出现数、输出token及新增token。自然结束且有boxed是格式指标，不保证答案正确；正文中的视觉反思措辞也不证明重新看图。

流式记录给出首次收到达到8k/16k token数的chunk的时间。这是**客户端观测、受chunk粒度影响的近似时间上界**，不是两次独立请求的耗时；额外token可精确计算。整次运行的墙钟时间另记controller session，不把估计时间冒充平台账单。

组员B复核：有效继续推理、算出候选答案但没有收束、重复检查/循环、视觉读图错误、答案提取问题。尚未完成人工标注。

本轮完毕再判断是否做同题100题16k；如16k继续重复，应单独设计答案整理阶段。不会自动加到32k、修改提示词、加入重复惩罚、挑更好回答或换掉难题。仍需使用同批100题做扩展并披露这是开发后的复评。

方法来源：[vLLM 0.11.0流式token IDs](https://docs.vllm.ai/en/v0.11.0/serving/openai_compatible_server.html)、[输出预算与EOS参数](https://docs.vllm.ai/en/v0.11.0/api/vllm/sampling_params.html)。
