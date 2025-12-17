"""multi_evaluate.py

Run evaluate.py repeatedly and aggregate winrate + foul breakdown.

Why this exists:
- The environment is stochastic (noise + randomized search), and evaluate.py defaults to
  *non-deterministic* execution.
- Single-run numbers swing a lot; averaging across runs makes tuning decisions clearer.

Usage examples:
- Quick sanity (3 runs x 30 games, deterministic seeds):
  python multi_evaluate.py --runs 3 --games 30 --seed 42

- More stable estimate (5 runs x 100 games):
  python multi_evaluate.py --runs 5 --games 100 --seed 42

Notes:
- This script suppresses evaluate.py's verbose per-shot printing.
- Each run's evaluation_log.json is archived into --outdir.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


@dataclass(frozen=True)
class RunSummary:
    seed: int | None
    games: int
    win_rate_b: float
    b_total_shots: int
    b_total_fouls: int
    b_foul_breakdown: dict[str, int]


def _load_eval(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summarize(eval_json: dict[str, Any], games: int, seed: int | None) -> RunSummary:
    results = eval_json.get("results", {})
    foul_stats = eval_json.get("foul_stats", {})

    # evaluate.py uses SCORE (win + 0.5*tie) / n_games
    win_rate_b = float(results.get("AGENT_B_SCORE", 0.0)) / float(games)

    b = foul_stats.get("AGENT_B", {})
    b_total_shots = int(b.get("total_shots", 0))
    breakdown = {
        "cue_pocket": int(b.get("cue_pocket", 0)),
        "eight_illegal": int(b.get("eight_illegal", 0)),
        "first_foul": int(b.get("first_foul", 0)),
        "rail_foul": int(b.get("rail_foul", 0)),
        "no_hit": int(b.get("no_hit", 0)),
    }
    b_total_fouls = int(sum(breakdown.values()))

    return RunSummary(
        seed=seed,
        games=games,
        win_rate_b=win_rate_b,
        b_total_shots=b_total_shots,
        b_total_fouls=b_total_fouls,
        b_foul_breakdown=breakdown,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluate.py multiple times and aggregate results")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed. If provided, run i uses seed+ i and evaluate.py becomes deterministic.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show evaluate.py output (useful for debugging failures)",
    )
    parser.add_argument("--outdir", type=str, default="eval_runs")
    args = parser.parse_args()

    runs = int(args.runs)
    games = int(args.games)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    summaries: list[RunSummary] = []

    for i in range(runs):
        run_seed = None if args.seed is None else int(args.seed) + i

        cmd = [sys.executable, "evaluate.py", "--games", str(games)]
        if run_seed is not None:
            cmd += ["--seed", str(run_seed)]

        if args.verbose:
            completed = subprocess.run(cmd, env=os.environ.copy())
            if completed.returncode != 0:
                print(f"Run {i+1}/{runs} failed (exit={completed.returncode}).", file=sys.stderr)
                print(f"Command: {' '.join(cmd)}", file=sys.stderr)
                return completed.returncode
        else:
            # Suppress the extremely verbose per-shot output, but keep stderr for debugging.
            completed = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy(),
            )
            if completed.returncode != 0:
                print(f"Run {i+1}/{runs} failed (exit={completed.returncode}).", file=sys.stderr)
                print(f"Command: {' '.join(cmd)}", file=sys.stderr)
                if completed.stderr:
                    tail = completed.stderr[-4000:]
                    print("---- evaluate.py stderr (tail) ----", file=sys.stderr)
                    print(tail, file=sys.stderr)
                    print("----------------------------------", file=sys.stderr)
                return completed.returncode

        eval_path = Path("evaluation_log.json")
        if not eval_path.exists():
            print("evaluation_log.json not found after run", file=sys.stderr)
            return 2

        eval_json = _load_eval(eval_path)
        summaries.append(_summarize(eval_json, games=games, seed=run_seed))

        archived = outdir / f"evaluation_log_{timestamp}_run{i+1:02d}_games{games}_seed{run_seed}.json"
        shutil.copyfile(eval_path, archived)

    winrates = [s.win_rate_b for s in summaries]

    def _sum_key(k: str) -> list[int]:
        return [s.b_foul_breakdown[k] for s in summaries]

    print("=" * 60)
    print(f"Aggregated over {runs} runs x {games} games")
    if args.seed is not None:
        print(f"Seeds: {args.seed}..{args.seed + runs - 1}")
    print(f"Logs archived to: {outdir}")
    print("=" * 60)

    print(f"NewAgent winrate: mean={mean(winrates)*100:.1f}%  std={pstdev(winrates)*100:.1f}%")

    # Foul per-game averages
    for key, label in [
        ("cue_pocket", "cue scratch"),
        ("eight_illegal", "illegal 8"),
        ("first_foul", "first-contact foul"),
        ("rail_foul", "no-pocket-no-rail"),
        ("no_hit", "no-hit"),
    ]:
        vals = _sum_key(key)
        print(f"{label:18s}: avg={mean(vals):.2f} per {games} games  (min={min(vals)}, max={max(vals)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
