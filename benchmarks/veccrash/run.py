"""VecCrash driver: kill a writer at seeded random times, verify, tally.

Example (the command used for results.json is recorded in it):

    python benchmarks/veccrash/run.py --trials 1000 --workers 3

For each library: three uncut calibration runs measure how long the writer
takes after its ready signal; each trial then SIGKILLs the writer at a
uniform random delay in [0, 1.1 x that time], so kills land across adds
and durability points (a few land after the writer finished). The store is
then verified in a fresh process (verify.py).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from adapters import ADAPTERS
from env import environment, wilson

HERE = Path(__file__).resolve().parent
OUTCOMES = ["OK", "EXTRA_PARTIAL", "LOST_ACKED", "SILENT_CORRUPTION", "CORRUPT_OPEN"]


def common(a, lib: str, d: Path, seed: int) -> list[str]:
    return [
        "--lib",
        lib,
        "--dir",
        str(d / "store"),
        "--ack",
        str(d / "ack.log"),
        "--seed",
        str(seed),
        "--dim",
        str(a.dim),
        "--batch",
        str(a.batch),
        "--batches",
        str(a.batches),
    ]


def spawn_writer(a, lib: str, d: Path, seed: int) -> subprocess.Popen:
    cmd = [sys.executable, str(HERE / "writer.py"), *common(a, lib, d, seed)]
    cmd += ["--ready", str(d / "ready")]
    return subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True
    )


def wait_ready(p: subprocess.Popen, d: Path, timeout: float = 120) -> None:
    end = time.monotonic() + timeout
    while not (d / "ready").exists():
        if p.poll() is not None or time.monotonic() > end:
            err = p.stderr.read().decode(errors="replace")[-800:] if p.stderr else ""
            raise RuntimeError(f"writer failed before ready: {err}")
        time.sleep(0.001)


def calibrate(a, lib: str) -> float:
    times = []
    for i in range(3):
        d = Path(a.workdir) / f"cal-{lib}-{i}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        p = spawn_writer(a, lib, d, 10_000 + i)
        wait_ready(p, d)
        t0 = time.monotonic()
        p.wait()
        times.append(time.monotonic() - t0)
        shutil.rmtree(d)
    return sorted(times)[1]


def verify(a, lib: str, d: Path, seed: int) -> dict:
    cmd = [sys.executable, str(HERE / "verify.py"), *common(a, lib, d, seed)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=a.verify_timeout)
    except subprocess.TimeoutExpired:
        return {"outcome": "CORRUPT_OPEN", "error": "verifier hung (timeout)"}
    if r.returncode != 0:
        tail = r.stderr.decode(errors="replace")[-300:]
        return {
            "outcome": "CORRUPT_OPEN",
            "error": f"verifier exit {r.returncode}: {tail}",
        }
    return json.loads(r.stdout.decode().strip().splitlines()[-1])


def trial(a, lib: str, i: int, window: float) -> dict:
    rng = random.Random(f"{a.seed}:{lib}:{i}")
    seed = rng.randrange(1 << 31)
    delay = rng.uniform(0, 1.1 * window)
    d = Path(a.workdir) / f"{lib}-{i}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    t0 = time.monotonic()
    p = spawn_writer(a, lib, d, seed)
    wait_ready(p, d)
    time.sleep(delay)
    finished = p.poll() is not None
    if not finished:
        os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    res = verify(a, lib, d, seed)
    res.update(
        trial=i,
        seed=seed,
        kill_delay_s=round(delay, 5),
        writer_finished_before_kill=finished,
        trial_s=round(time.monotonic() - t0, 3),
    )
    shutil.rmtree(d, ignore_errors=True)
    return res


def summarize(lib: str, window: float, rows: list[dict]) -> dict:
    n = len(rows)
    counts = {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES}
    examples = {}
    for o in OUTCOMES[1:]:
        ex = [r for r in rows if r["outcome"] == o][:5]
        if ex:
            examples[o] = ex
    trial_s = sorted(r["trial_s"] for r in rows)
    return {
        "trials": n,
        "counts": counts,
        "wilson95": {o: wilson(c, n) for o, c in counts.items()},
        "violations": counts["LOST_ACKED"]
        + counts["SILENT_CORRUPTION"]
        + counts["CORRUPT_OPEN"],
        "writer_finished_before_kill": sum(
            r["writer_finished_before_kill"] for r in rows
        ),
        "calibrated_window_s": round(window, 4),
        "median_trial_s": trial_s[n // 2],
        "examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", default=",".join(ADAPTERS))
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--verify-timeout", type=float, default=120)
    ap.add_argument("--workdir", default="/data/build/veccrash/work")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    a = ap.parse_args()

    out = Path(a.out)
    results = json.loads(out.read_text()) if out.exists() else {"libraries": {}}
    results["config"] = {
        k: getattr(a, k) for k in ("trials", "seed", "dim", "batch", "batches")
    }
    results["command"] = "python " + " ".join(sys.argv)
    results["environment"] = environment(a.workdir)
    trials_dir = out.parent / "trials"
    trials_dir.mkdir(exist_ok=True)
    for lib in a.libs.split(","):
        window = calibrate(a, lib)
        print(f"{lib}: window {window:.3f}s", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(partial(trial, a, lib, window=window), range(a.trials)))
        with open(trials_dir / f"{lib}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        results["libraries"][lib] = summarize(lib, window, rows)
        print(f"{lib}: {results['libraries'][lib]['counts']}", flush=True)
        out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
