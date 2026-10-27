# Sources and versions

The English edition is based on the rolled-back original reproduction archive, with the added GPU entry points removed. The later same-20-question Qwen comparison is not included. Original team code, raw responses, question images and numerical evidence remain byte-identical.

| Source | Location | Purpose |
|---|---|---|
| Original Qwen/VL project | `pipelines/qwen_vl/` | Original 20 plus 80 Qwen runs, VL100, inference code and custom review |
| Teammate OVR project | `pipelines/ovr/` | OVR100, budget20, teacher traces, tokenizer, official evaluator and tests |
| Additional original scripts | `sources/user_project_scripts/` | Historical versions for provenance |
| Original Word reports | `sources/team_reports/` | Unchanged source-language reports |
| English presentation | `START_HERE.html`, `docs/`, `README.md` | Current demonstration and interpretation |
| English results wrapper | `scripts/reproduce_english.py` | Runs the unchanged original pipeline, then writes an English summary |

## Upstream resources

- OVR: https://github.com/Open-Reasoner-Zero/Open-Vision-Reasoner
- ColdStart: https://huggingface.co/Kangheng/OVR-7B-ColdStart ; revision `52ad9cb58c52d8e5f8c18abee99f77f78e9f6be3`
- RL: https://huggingface.co/Kangheng/OVR-7B-RL ; revision `53d5680f59196df1533ca5ec55d9046507c1e6d7`
- Teacher data: https://huggingface.co/datasets/Kangheng/OVR-Dataset-ColdStart ; revision `e377de8ec93f9d1ff48744dcdc8c51c4801c6da2`
- VL-Rethinker: https://huggingface.co/TIGER-Lab/VL-Rethinker-7B ; revision `5439d11fc9fa1474ab015104e6604223f165ccf5`
- Qwen: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct ; the original download revision was not recorded, which limits exact inference reproduction.
- MathVision: https://github.com/mathllm/MATH-V ; data and scorer revision `a1f5cc3add200c0cd080fad463e500f44ef1fb41`

The archived evaluator and license are in `pipelines/ovr/vendor/mathvision/`. File hashes and retrieval records are in the original source manifests. Original image licenses are retained with the datasets. No new blanket license overrides upstream material.

Teacher sampling uses five recorded Parquet shards totaling 1,637,924 rows. A partial web viewer is not the sampling population. The frozen 1,000-row sample and common tokenizer are included; the full 11.78GB teacher dataset is not required for offline reproduction.

The final distribution manifest covers the English additions and unchanged originals. Historical cloud paths, execution windows and instructions are evidence of earlier work, not current runtime directions. No new model generation, training or cloud instance operation was performed while translating this edition.
