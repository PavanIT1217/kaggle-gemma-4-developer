#!/usr/bin/env python3
"""Build (and optionally push) the Kaggle eval notebook with submission/ and a task list embedded.

    python scripts/build_eval_kernel.py --tasks rich_3718,fastapi_1234      # specific tasks
    python scripts/build_eval_kernel.py --tasks-file eval/dev.txt --push     # one id per line
    python scripts/build_eval_kernel.py --push                               # all 129 public tasks (long!)

Writes build/agent_eval/. --push runs `kaggle kernels push` (needs ~/.kaggle/kaggle.json).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submission")
    ap.add_argument("--tasks", default="", help="comma-separated task ids")
    ap.add_argument("--tasks-file", help="file with one task id per line")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    sub = ROOT / args.submission
    if subprocess.run([sys.executable, str(ROOT / "scripts/validate_submission.py"), str(sub)]).returncode:
        return 1
    files = {str(p.relative_to(sub)): p.read_text(encoding="utf-8")
             for p in sorted(sub.rglob("*")) if p.is_file() and p.suffix != ".safetensors"}
    if any(sub.rglob("*.safetensors")):
        print("note: LoRA adapters are not embedded; upload them as a dataset for eval runs")
    tasks = [t for t in args.tasks.split(",") if t]
    if args.tasks_file:
        tasks += [l.strip() for l in Path(args.tasks_file).read_text().splitlines() if l.strip()]

    src = ROOT / "kaggle/agent_eval"
    out = ROOT / "build/agent_eval"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copy(src / "kernel-metadata.json", out)
    code = (src / "agent_eval.py").read_text()
    code = code.replace("SUBMISSION_FILES: dict[str, str] = {}", f"SUBMISSION_FILES: dict[str, str] = {json.dumps(files)}", 1)
    code = code.replace("TASK_IDS: list[str] = []", f"TASK_IDS: list[str] = {json.dumps(tasks)}", 1)
    (out / "agent_eval.py").write_text(code)
    print(f"built {out} with {len(files)} config files, {len(tasks) or 'all'} tasks")

    if args.push:
        meta = json.loads((out / "kernel-metadata.json").read_text())
        if meta["id"].startswith("YOUR_KAGGLE_USERNAME"):
            print("error: set your username in kaggle/agent_eval/kernel-metadata.json first")
            return 1
        return subprocess.run(["kaggle", "kernels", "push", "-p", str(out)]).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
