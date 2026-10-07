# Local development and evaluation

The official harness (`swegemma`, `adk-submission`) and the model are only distributed on Kaggle,
and the model needs ~48 GB of Ampere-or-newer GPU memory. So the loop has two layers:

| Layer | Where | GPU | Time | What it tells you |
|---|---|---|---|---|
| 1. Validate | anywhere (`python scripts/validate_submission.py`) | no | 1 s | the zip won't be rejected or blow the 12 h limit |
| 2. Agent eval | Kaggle notebook, 4×L4 (`scripts/build_eval_kernel.py`) | yes | ~20 min + ~5 min/task | how many public tasks the agent solves, and full traces of *why* |
| 3. Submit | Kaggle | — | hours, 1/day | the real (public-leaderboard) score |

## One-time setup

1. On Kaggle: join the competition, accept the rules, and verify your phone number (needed for GPU).
2. Kaggle → Settings → API → *Create New Token*; put `kaggle.json` in `~/.kaggle/` (`chmod 600`).
3. `pip install kaggle pyyaml`
4. Set your username in `kaggle/agent_eval/kernel-metadata.json` (`"id": "<you>/gemma-agent-eval"`).
5. Read `HARNESS_README.md` on the competition's Data tab.

## The loop

```bash
# 1. edit submission/ (prompt, sampling, budgets, sub-agents...)
python scripts/validate_submission.py

# 2. run it on a handful of public tasks on Kaggle's 4xL4 (same hardware as the scorer)
python scripts/build_eval_kernel.py --tasks-file eval/dev.txt --push
kaggle kernels status <you>/gemma-agent-eval            # QUEUED / RUNNING / COMPLETE / ERROR
kaggle kernels output <you>/gemma-agent-eval -p eval/runs/$(date +%m%d-%H%M)

# 3. read eval/runs/<run>/run_summary.json and the traces in results/traces/ for failures
# 4. if it's better, ./scripts/pack_submission.sh and submit submission.zip
```

You can also do step 2 by hand: create a notebook on the competition page, choose the GPU L4×4
accelerator, add the `metric/gemma-4-developer-agent-wheelhouse` dataset and the
`gemma-4-31b-it-qat-w4a16-ct` model as inputs, and paste in `build/agent_eval/agent_eval.py`.

## Picking dev tasks

There are 129 public tasks (fastapi 67, rich 48, requests 13, httpx 1). Find their ids in
`tasks.jsonl` on the Data tab and put 10–30 of them in `eval/dev.txt`, one per line. Use a
mix of repos, and prefer the newest issues because they're less likely to be in the model's
training data. Keep this set fixed so runs stay comparable.

## Reading results honestly

- **Local scores run high.** Other teams got 0.18–0.24 on public tasks against 0.05–0.12 on the
  leaderboard, because the public repos are famous and the hidden ones are private.
  Use local runs to compare configs and to find failure modes, not to predict the leaderboard.
- **Noise is large.** One hidden task is ~0.017 on the public leaderboard, and identical configs
  have scored 0.08 vs 0.12. Change one thing at a time and log each run in a table.
- **Read the traces of failed runs.** Common failures: context overflow (scores 0 even after
  `submit_patch`), running out of time before editing, wrong file, editing tests, and a fix
  that doesn't use the exact names from the issue.
- **The notebook isn't an exact copy of the scorer.** It uses the `subprocess` sandbox, with no
  Docker and the notebook's own Python packages, so the odd task may pass or fail for
  environment reasons.
- **GPU quota.** 4×L4 costs 2× quota. Test with 2–3 tasks before launching a long run.

## Without a GPU

You can still step through the agent loop on CPU by pointing the harness at a mock
OpenAI-compatible server that replays scripted tool calls. This tests plumbing only, not
problem-solving. It's useful when writing skills or sub-agents. See
[LogosTopos/gemma-4-developer-agent](https://github.com/LogosTopos/gemma-4-developer-agent)
(`scripts/mock_llm_server.py`) for an example.
