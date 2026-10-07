#!/usr/bin/env python3
"""Check submission/ against the competition rules without a GPU or the official harness.

Catches the mistakes that waste a daily submission: wrong model name, broken !include paths,
disallowed files, missing or over-budget eval_config.yaml, output cap above the limit.

    python scripts/validate_submission.py            # checks ./submission
    python scripts/validate_submission.py path/to/dir
"""
import sys
from pathlib import Path

import yaml

MODEL = "gemma-4-31b-it-qat-w4a16-ct"
ALLOWED_SUFFIXES = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
HARNESS_TOOLS = {
    "run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch",
    "search_similar_code", "get_code_neighbors", "get_code_subgraph",
    "run_skill_script", "load_skill_resource", "AgentTool",
}
EVAL_KEYS = {"max_time_minutes", "max_tool_calls", "max_turns", "timeout_seconds"}
MAX_OUTPUT_TOKENS = 32768
MAX_ZIP_BYTES = 3 * 1024**3
HIDDEN_TASKS = 120
SETUP_MINUTES = 1.0       # rough per-task sandbox setup, counted against the 12 h limit
RUN_LIMIT_HOURS = 12.0

errors: list[str] = []
warnings: list[str] = []


def load(path: Path, root: Path):
    """Load YAML, resolving `!include` relative to the including file."""

    class Loader(yaml.SafeLoader):
        pass

    def include(loader, node):
        target = (path.parent / loader.construct_scalar(node)).resolve()
        if not target.is_relative_to(root):
            errors.append(f"{path.relative_to(root)}: !include escapes the submission: {node.value}")
            return None
        if not target.is_file():
            errors.append(f"{path.relative_to(root)}: !include target missing: {node.value}")
            return None
        if target.suffix in {".yaml", ".yml"}:
            return load(target, root)
        return target.read_text(encoding="utf-8")

    Loader.add_constructor("!include", include)
    return yaml.load(path.read_text(encoding="utf-8"), Loader)


def check_agent(path: Path, root: Path, seen: set[Path]) -> None:
    if path in seen:
        return
    seen.add(path)
    rel = path.relative_to(root)
    cfg = load(path, root) or {}
    if cfg.get("model") != MODEL:
        errors.append(f"{rel}: model must be {MODEL!r}, got {cfg.get('model')!r}")
    if not cfg.get("name"):
        errors.append(f"{rel}: missing name")
    if not cfg.get("instruction"):
        warnings.append(f"{rel}: no instruction (system prompt)")

    gen = cfg.get("generate_content_config") or {}
    cap = gen.get("max_output_tokens")
    if cap is not None and cap > MAX_OUTPUT_TOKENS:
        errors.append(f"{rel}: max_output_tokens {cap} > {MAX_OUTPUT_TOKENS}")
    if (gen.get("thinking_config") or {}).get("include_thoughts"):
        warnings.append(f"{rel}: thinking is on; every 0.10+ public config turns it off")

    names = []
    for tool in cfg.get("tools") or []:
        name = tool if isinstance(tool, str) else (tool or {}).get("name")
        names.append(name)
        if name not in HARNESS_TOOLS:
            warnings.append(f"{rel}: unknown tool {name!r} (not a harness tool)")
        if name == "AgentTool":
            sub = ((tool.get("args") or {}).get("agent") or {}).get("config_path")
            sub_path = (path.parent / str(sub)).resolve()
            if not sub or not sub_path.is_file():
                errors.append(f"{rel}: AgentTool config_path not found: {sub}")
            else:
                check_agent(sub_path, root, seen)
    for sub in cfg.get("sub_agents") or []:
        sub_path = (path.parent / str((sub or {}).get("config_path"))).resolve()
        if sub_path.is_file():
            check_agent(sub_path, root, seen)
        else:
            errors.append(f"{rel}: sub_agent config_path not found: {sub}")
    if path.name == "agent.yaml" and path.parent == root and "submit_patch" not in names:
        warnings.append(f"{rel}: root agent has no submit_patch (the diff is still taken, but the agent can't end early)")


def check_budget(root: Path) -> None:
    path = root / "eval_config.yaml"
    if not path.is_file():
        errors.append("eval_config.yaml missing: defaults are 60 min/task, which blows the 12 h limit")
        return
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = raw.get("evaluation", raw)
    unknown = set(section) - EVAL_KEYS
    if unknown:
        warnings.append(f"eval_config.yaml: keys the scorer ignores: {sorted(unknown)}")
    minutes = float(section.get("max_time_minutes", 60))
    worst = HIDDEN_TASKS * (minutes + SETUP_MINUTES) / 60
    line = f"worst case {HIDDEN_TASKS} x ({minutes} + ~{SETUP_MINUTES:g} setup) min = {worst:.1f} h of {RUN_LIMIT_HOURS:g} h"
    if worst > RUN_LIMIT_HOURS:
        errors.append(f"eval_config.yaml: {line}")
    elif worst > RUN_LIMIT_HOURS - 0.75:
        warnings.append(f"eval_config.yaml: {line} (little headroom)")
    else:
        print(f"budget: {line}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "submission").resolve()
    if not (root / "agent.yaml").is_file():
        print(f"ERROR: {root}/agent.yaml not found (it must be at the zip root)")
        return 1

    total = 0
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if p.is_symlink():
            errors.append(f"{rel}: symlinks are not allowed")
        elif p.is_file():
            total += p.stat().st_size
            if p.suffix not in ALLOWED_SUFFIXES:
                errors.append(f"{rel}: file type {p.suffix or '(none)'} not allowed")
    if total > MAX_ZIP_BYTES:
        errors.append(f"submission is {total / 1024**3:.2f} GiB unpacked (limit 3 GiB)")

    check_agent(root / "agent.yaml", root, set())
    check_budget(root)

    for w in warnings:
        print(f"WARN:  {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
