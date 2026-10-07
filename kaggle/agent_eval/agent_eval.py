"""Run our agent config on public training tasks with the official harness, like the scorer does.

Runs on a Kaggle notebook attached to the competition with 4x L4 GPUs: the scorer's model
(gemma-4-31b-it-qat-w4a16-ct served by vLLM), its harness (swegemma), our eval_config.yaml
budgets, and tasks one after another. Don't run this file directly: scripts/build_eval_kernel.py
embeds submission/ and the task list into a copy under build/ and pushes that.

Outputs in /kaggle/working:
    results/           harness output: summary.json, task_results.jsonl, patches/, traces/, logs/
    run_summary.json   one row per task: resolved, error, tool calls, minutes, patch size
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DATA_DIR = Path("/kaggle/input/competitions/gemma-4-developer-agent")
WHEELHOUSE = Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse")
MODEL_PATH = Path("/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2")
OUT_DIR = Path("/kaggle/working")
MODEL = "gemma-4-31b-it-qat-w4a16-ct"

SUBMISSION_FILES: dict[str, str] = {}  # filled in by build_eval_kernel.py
TASK_IDS: list[str] = []  # filled in by build_eval_kernel.py; empty = all public tasks


def find(default: Path, marker: str) -> Path:
    """Kaggle mount paths drift between versions: use `default` if present, else the folder
    under /kaggle/input that holds `marker`."""
    if default.exists():
        return default
    hits = sorted(Path("/kaggle/input").rglob(marker))
    if not hits:
        sys.exit(f"Could not find {marker} under /kaggle/input; attach the data source.")
    return hits[0].parent


def install_harness(wheelhouse: Path) -> None:
    tmp = Path("/tmp/wheelhouse")
    tmp.mkdir(exist_ok=True)
    for whl in wheelhouse.rglob("*.whl"):
        # Kaggle drops '+' from uploaded file names; restore the local version tag.
        name = whl.name.replace("cu128", "+cu128") if "cu128" in whl.name and "+" not in whl.name else whl.name
        if not (tmp / name).exists():
            os.symlink(whl, tmp / name)
    wheels = sorted(str(w) for w in tmp.glob("*.whl"))
    print(f"Installing {len(wheels)} harness wheels", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *wheels], check=True)


def main() -> None:
    for key, value in {
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
        "VLLM_NO_USAGE_STATS": "1",
        "OTEL_SDK_DISABLED": "true",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }.items():
        os.environ.setdefault(key, value)

    data_dir = find(DATA_DIR, "tasks.jsonl")
    install_harness(find(WHEELHOUSE, "swegemma*.whl"))
    model_path = find(MODEL_PATH, "config.json")

    agent_dir = OUT_DIR / "agent"
    for rel, text in SUBMISSION_FILES.items():
        (agent_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (agent_dir / rel).write_text(text, encoding="utf-8")

    import litellm
    import torch
    import yaml
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator

    litellm.drop_params = True
    adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    gpus = torch.cuda.device_count()
    # Server settings from the organizers' getting-started notebook (= scorer hardware).
    server = VllmServer(VllmConfig(
        model=str(model_path), host="127.0.0.1", port=8000,
        tool_call_parser="gemma4", reasoning_parser="gemma4",
        max_model_len=32768, dtype="bfloat16", gpu_memory_utilization=0.90,
        enable_auto_tool_choice=True, enable_lora=True, max_loras=8, max_lora_rank=128,
        tensor_parallel_size=4 if gpus >= 4 else max(gpus, 1), startup_timeout=60 * 20,
    ), adapter_manifest=adapters)
    t0 = time.time()
    server.start()
    print(f"vLLM ready after {(time.time() - t0) / 60:.1f} min on {gpus} GPUs", flush=True)
    models = server.create_model_registry(aliases=[MODEL], model_prefix="openai/", api_key="EMPTY")

    eval_path = agent_dir / "eval_config.yaml"
    raw = yaml.safe_load(eval_path.read_text()) if eval_path.exists() else {}
    budget = (raw or {}).get("evaluation", raw or {})
    limits, gen_constraints = build_submission_limits()
    config = EvalConfig(
        tasks_path=data_dir / "tasks.jsonl",
        snapshots_dir=data_dir / "snapshots",
        results_dir=OUT_DIR / "results",
        submission_dir=agent_dir,
        models=models,
        sandbox="subprocess",  # no Docker in notebooks; tasks share the notebook's packages
        timeout_seconds=int(budget.get("timeout_seconds", 300)),
        max_time_minutes=float(budget.get("max_time_minutes", 60.0)),
        max_tool_calls=int(budget.get("max_tool_calls", 100)),
        max_turns=int(budget["max_turns"]) if budget.get("max_turns") else None,
        limits=limits,
        generation_constraints=gen_constraints,
        adapter_manifest=adapters,
        graph_dir=str(data_dir / "graphs"),
        embeddings_dir=str(data_dir / "embeddings"),
        wheels_dir=data_dir / "wheels",
        task_ids=TASK_IDS or None,
        concurrency=1,  # the scorer runs tasks one at a time
        display_mode="quiet",
    )
    print(f"Running {len(TASK_IDS) or 'all'} tasks with budget {dict(budget)}", flush=True)
    t0 = time.time()
    result = asyncio.run(Evaluator(config).run())

    rows = []
    for r in sorted(result.task_results, key=lambda r: r.instance_id):
        rows.append({
            "instance_id": r.instance_id,
            "resolved": bool(r.resolved),
            "error": (r.error or "")[:300],
            "tool_calls": r.tool_calls,
            "minutes": round((r.duration_seconds or 0) / 60, 2),
            "patch_chars": len(r.agent_patch or ""),
        })
        print(f"{r.instance_id:20} {'PASS' if r.resolved else 'fail'}  calls={r.tool_calls:<3} "
              f"{rows[-1]['minutes']:.1f} min  {(r.error or '')[:80]}", flush=True)
    summary = {"resolved": result.resolved, "total": result.total,
               "hours": round((time.time() - t0) / 3600, 2), "tasks": rows}
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResolved {result.resolved}/{result.total} in {summary['hours']} h")


if __name__ == "__main__":
    main()
