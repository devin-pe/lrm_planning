# Optimal And Solved Results by Number of Disks

Source folders:
- `agent_results/deepseek-r1-tower`
- `agent_results/gpt-oss-tower`
- `agent_results/kimi-k2-think-tower`
- `agent_results/llama4-tower`

Counting rule:
- `optimal`: `validation.solved == true` and `validation.is_optimal == true`
- `solved`: `validation.solved == true` (includes both optimal and suboptimal)

Cell format:
- `optimal/10 (solved/10)`

| Model | n=3 | n=4 | n=5 |
|---|---:|---:|---:|
| deepseek-r1-tower | 2/10 (2/10) | 3/10 (3/10) | 4/10 (4/10) |
| gpt-oss-tower | 6/10 (6/10) | 1/10 (1/10) | 3/10 (3/10) |
| kimi-k2-think-tower | 8/10 (8/10) | 10/10 (10/10) | 10/10 (10/10) |
| llama4-tower | 3/10 (3/10) | 2/10 (2/10) | 2/10 (2/10) |

Note:
- `agent_results/deepseek-r1-tower` currently contains 14 `problem_*.json` files (n=3: 5, n=4: 3, n=5: 6), so its `/10` cells are normalized to your requested denominator rather than observed counts.

Generated on: 2026-03-05
